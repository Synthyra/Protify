from . import entrypoint_setup


# Hugging Face reads its settings when it loads, so they are set before anything below imports it.
entrypoint_setup.configure_process()

import os
import math
import time
import torch
import torch.multiprocessing as mp
import warnings
import gzip
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Iterator, Sequence
from huggingface_hub import hf_hub_download

from .seed_utils import seed_worker, dataloader_generator, get_global_seed
from .data.dataset_classes import SimpleProteinDataset
from .base_models.get_base_models import get_base_model
from .pooler import Pooler
from .console_output import print_message
from .feature_cache import (
    CacheWriter, carbon_cache_identity, embedding_descriptor, embedding_layout, has_feature,
    is_generated_model, legacy_conversion_refusal, locate_feature, missing_sequences, read_embeddings,
)
from .legacy_cache import convert_legacy_cache


# Truncation limit every embedding in the download repository was produced under, and the
# default everywhere a max_length is not supplied.
PUBLISHED_EMBEDDING_MAX_LENGTH = 2048


def maybe_compile(model: torch.nn.Module, dynamic: bool = False) -> torch.nn.Module:
    if dynamic:
        # dynamic=True (padding='longest') is incompatible with flex attention's
        # create_block_mask under torch.compile, causing CUDA illegal memory access.
        # Skip compilation; the variable-shape batches already avoid wasted padding.
        print_message("Skipping torch.compile (dynamic shapes + flex attention incompatible)")
        return model
    try:
        model = torch.compile(model)
        print_message("Model compiled")
    except Exception as e:  # noqa: broad-except torch.compile can fail in any compiler layer; the model then runs uncompiled
        print_message(f"Skipping torch.compile: {e}")
    return model


def build_collator(tokenizer: object, padding: str = 'max_length', max_length: int = 2048) -> Callable[[list[str]], dict[str, torch.Tensor]]:
    def _collate_fn(sequences: list[str]) -> dict[str, torch.Tensor]:
        """Collate function for batching sequences."""
        kwargs: dict[str, Any] = dict(
            return_tensors="pt", padding=padding, truncation=True, max_length=max_length,
        )
        if padding != 'max_length':
            kwargs['pad_to_multiple_of'] = 8
        return tokenizer(sequences, **kwargs)  # input_ids and attention_mask, each (b, n): n is the longest l + 2 (CLS, residues, EOS) plus padding
    return _collate_fn  # sequences -> input_ids, attention_mask: (b, n)


def get_embedding_filename(
        model_name: str,
        matrix_embed: bool,
        pooling_types: list[str],
        extension: str = 'pth',
        hidden_state_index: int = -1,
        max_length: int = 2048,
        model_type: str | None = None,
        model_path: str | None = None,
) -> str:
    """Name the embedding cache file after every input that can change the stored values.

    Protify keeps its embeddings in the feature store now, under the key of
    `feature_cache.embedding_descriptor`. This is the name of the file an earlier version wrote, which
    `legacy_cache` converts and `_download_embeddings` asks the Hub for.

    Args:
        model_name: Resolved display name of the model, which already carries SAE settings.
        matrix_embed: True for per-residue matrices, False for pooled vectors.
        pooling_types: Pooling operations, used only for vector embeddings.
        extension: 'pth' or 'db'.
        hidden_state_index: Hidden-state tuple index. -1 uses the final hidden state.
        max_length: Tokenizer truncation limit, which decides how much of a long
            sequence reached the model.
        model_type: Dispatch keyword when it differs from model_name, as in path mode.
        model_path: Explicit path in path mode, needed to resolve a CARBON revision.
    """
    assert isinstance(hidden_state_index, int), "hidden_state_index must be an integer."
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    base_name = f'{model_name}_{matrix_embed}'
    if hidden_state_index != -1:
        base_name = f'{base_name}_hs{hidden_state_index}'
    base_name = f'{base_name}_len{max_length}'

    carbon_identity = carbon_cache_identity(model_type or model_name, model_path)
    if carbon_identity is not None:
        revision, preprocessing = carbon_identity
        base_name = f'{base_name}_rev{revision}_prep{preprocessing}'
    if not matrix_embed and pooling_types:
        # For vector embeddings, include pooling types in filename
        pooling_str = '_'.join(sorted(pooling_types))  # Sort for consistency
        base_name = f'{base_name}_{pooling_str}'
    return f'{base_name}.{extension}'


def legacy_embedding_filename(
        model_name: str,
        matrix_embed: bool,
        pooling_types: list[str],
        extension: str = 'pth',
        hidden_state_index: int = -1,
) -> str:
    """Name a cache the way Protify did before max_length joined the cache identity.

    The published embeddings in the download repository were uploaded under this scheme,
    all of them produced at PUBLISHED_EMBEDDING_MAX_LENGTH. Only `_download_embeddings`
    should call this, so a stale remote name never becomes the name we write locally.
    """
    base_name = f'{model_name}_{matrix_embed}'
    if hidden_state_index != -1:
        base_name = f'{base_name}_hs{hidden_state_index}'
    if not matrix_embed and pooling_types:
        base_name = f'{base_name}_{"_".join(sorted(pooling_types))}'
    return f'{base_name}.{extension}'


def _make_embedding_progress(
    dataloader: DataLoader,
    padding: str,
    n_warmup: int = 3,
    n_calibration: int = 5,
) -> Iterator[tuple[int, Any]]:
    """Progress-bar wrapper for embedding loops. Drop-in replacement for enumerate(dataloader).

    When padding='max_length', all batches have uniform cost so plain tqdm works.
    When padding='longest' (sorted longest-first), batch times vary dramatically.
    In that case: yield warmup batches first (compiler warmup + OOM check on longest
    sequences), then time mid-length calibration batches to estimate total ETA.

    Keep in sync with fastplms/embedding_mixin.py. Canonical source: core/embed/progress.py.
    """
    total = len(dataloader)
    if padding == 'max_length' or total <= n_warmup + n_calibration:
        yield from tqdm(enumerate(dataloader), total=total, desc='Embedding batches')
        return

    dl_iter = iter(dataloader)

    # Phase 1: warmup on longest batches (first n_warmup, since sorted longest-first)
    warmup_bar = tqdm(range(n_warmup), desc='Warmup (longest batches)', leave=False)
    for i in warmup_bar:
        batch = next(dl_iter)
        yield i, batch
    warmup_bar.close()

    # Phase 2: skip to middle of dataset for calibration timing
    # We need to yield all intermediate batches too (they contain real data)
    mid_start = total // 2
    intermediate_bar = tqdm(
        range(n_warmup, mid_start), desc='Embedding batches', leave=False,
    )
    for i in intermediate_bar:
        batch = next(dl_iter)
        yield i, batch
    intermediate_bar.close()

    # Phase 3: time calibration batches from the middle
    calibration_times: list[float] = []
    cal_bar = tqdm(range(n_calibration), desc='Calibrating ETA', leave=False)
    for j in cal_bar:
        t0 = time.perf_counter()
        batch = next(dl_iter)
        yield mid_start + j, batch
        calibration_times.append(time.perf_counter() - t0)
    cal_bar.close()

    avg_time = sum(calibration_times) / len(calibration_times)
    remaining_start = mid_start + n_calibration
    remaining_count = total - remaining_start
    estimated_total_seconds = avg_time * remaining_count

    # Phase 4: remaining batches with calibrated ETA
    main_bar = tqdm(
        range(remaining_count),
        desc='Embedding batches',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )
    main_bar.set_postfix_str(f'ETA ~{estimated_total_seconds:.0f}s (calibrated)')
    for k in main_bar:
        batch = next(dl_iter)
        yield remaining_start + k, batch
    main_bar.close()


@dataclass
class EmbeddingArguments:
    def __init__(
            self,
            embedding_batch_size: int = 4,
            embedding_num_workers: int = 0,
            download_embeddings: bool = False,
            download_dir: str = 'Synthyra/vector_embeddings',
            matrix_embed: bool = False,
            embedding_pooling_types: Sequence[str] | None = None,
            save_embeddings: bool = False,
            embed_dtype: torch.dtype = torch.float32,
            model_dtype: torch.dtype = None,
            sql: bool = False,
            embedding_save_dir: str = 'embeddings',
            padding: str = 'max_length',
            max_length: int = 2048,
            embedding_hidden_state_index: int = -1,
            multi_gpu: bool = False,
            autocast: bool = False,
            embedding_scaler: bool = True,
            **kwargs: Any
    ) -> None:
        self.batch_size = embedding_batch_size
        self.num_workers = embedding_num_workers
        self.download_embeddings = download_embeddings
        self.download_dir = download_dir
        self.matrix_embed = matrix_embed
        self.pooling_types = ['mean'] if embedding_pooling_types is None else list(embedding_pooling_types)
        self.save_embeddings = save_embeddings
        self.embed_dtype = embed_dtype
        self.model_dtype = model_dtype
        self.sql = sql
        self.embedding_save_dir = embedding_save_dir
        self.padding = padding
        self.max_length = max_length
        self.hidden_state_index = embedding_hidden_state_index
        self.multi_gpu = multi_gpu
        self.autocast = autocast
        assert isinstance(embedding_scaler, bool), f"Invalid embedding_scaler: {embedding_scaler}"
        self.embedding_scaler = embedding_scaler

    def cache_filename(
            self,
            model_name: str,
            matrix_embed: bool | None = None,
            extension: str = 'pth',
            model_type: str | None = None,
            model_path: str | None = None,
        ) -> str:
        return get_embedding_filename(
            model_name,
            self.matrix_embed if matrix_embed is None else matrix_embed,
            self.pooling_types,
            extension=extension,
            hidden_state_index=self.hidden_state_index,
            max_length=self.max_length,
            model_type=model_type,
            model_path=model_path,
        )

    def cache_descriptor(
            self,
            model_name: str,
            matrix_embed: bool | None = None,
            model_type: str | None = None,
            model_path: str | None = None,
        ) -> dict[str, Any]:
        """The plain data that names this run's cached representation of a model.

        Its digest is the store's key (`feature_cache.embedding_descriptor` lists the fields), so every
        writer and reader that builds the descriptor from the same arguments finds the same store.
        """
        return embedding_descriptor(
            model_name,
            matrix_embed=self.matrix_embed if matrix_embed is None else matrix_embed,
            pooling_types=self.pooling_types,
            hidden_state_index=self.hidden_state_index,
            max_length=self.max_length,
            dtype=self.embed_dtype,
            model_type=model_type,
            model_path=model_path,
        )

    def feature_directory(
            self,
            model_name: str,
            matrix_embed: bool | None = None,
            model_type: str | None = None,
            model_path: str | None = None,
        ) -> str:
        """The store that serves this run's cache of a model, whether or not it has been written yet.

        `embedding_save_dir` is the root that holds every model's store. A store that holds the run's
        dtype exactly, or a wider one, is found; otherwise this is where the run's own store will be made.
        """
        descriptor = self.cache_descriptor(model_name, matrix_embed, model_type, model_path)
        return str(locate_feature(self.embedding_save_dir, descriptor))

    def legacy_cache_files(
            self,
            model_name: str,
            matrix_embed: bool | None = None,
            model_type: str | None = None,
            model_path: str | None = None,
            extensions: tuple[str, ...] = ('db', 'pth'),
        ) -> list[str]:
        """The old-format cache files of this representation that exist in `embedding_save_dir`.

        `extensions` sets the order, so a caller can put the format its own mode used first.
        """
        names = (
            os.path.join(
                self.embedding_save_dir,
                self.cache_filename(model_name, matrix_embed, extension, model_type, model_path),
            )
            for extension in extensions
        )
        return [path for path in names if os.path.isfile(path)]


class Embedder:
    def __init__(self, args: EmbeddingArguments, all_seqs: list[str]) -> None:
        self.args = args
        self.all_seqs = all_seqs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.matrix_embed = args.matrix_embed
        self.pooling_types = args.pooling_types
        self.download_embeddings = args.download_embeddings
        self.download_dir = args.download_dir
        self.save_embeddings = args.save_embeddings
        self.embed_dtype = args.embed_dtype
        self.model_dtype = args.model_dtype
        self.sql = args.sql
        self.embedding_save_dir = args.embedding_save_dir
        self.padding = args.padding
        self.max_length = args.max_length
        self.hidden_state_index = args.hidden_state_index
        self.multi_gpu = args.multi_gpu
        self.autocast = args.autocast

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print_message(f'Device {self.device} found ({n_gpus} GPU{"s" if n_gpus != 1 else ""})')

    def _download_embeddings(
            self,
            model_name: str,
            model_type: str | None = None,
            model_path: str | None = None,
        ) -> str | None:
        # download from download_dir
        # unzip
        # convert into the feature store under embedding_save_dir
        filename = self.args.cache_filename(
            model_name,
            extension='pth',
            model_type=model_type,
            model_path=model_path,
        )
        descriptor = self.args.cache_descriptor(model_name, model_type=model_type, model_path=model_path)
        # Rows this machine already embedded go into the store first, so they win over the published ones.
        self._adopt_legacy_caches(model_name, descriptor, model_type, model_path)
        remote_names = [filename]
        # The published files predate max_length entering the cache identity. They are
        # still valid at the default limit, so ask for the old name too, and only then.
        # Below that limit the download would carry more of each sequence than this run
        # asked for, which is exactly what putting max_length in the name prevents.
        legacy_name = legacy_embedding_filename(
            model_name,
            self.matrix_embed,
            self.pooling_types,
            'pth',
            self.hidden_state_index,
        )
        if self.max_length == PUBLISHED_EMBEDDING_MAX_LENGTH and legacy_name != filename:
            remote_names.append(legacy_name)

        local_path = None
        for remote_name in remote_names:
            try:
                local_path = hf_hub_download(
                    repo_id=self.download_dir,
                    filename=f'embeddings/{remote_name}.gz',
                    repo_type='dataset'
                )
                break
            except Exception:  # noqa: broad-except any Hub failure for one candidate file name moves on to the next
                continue

        if local_path is None:
            print(f'No embeddings found for {model_name} in {self.download_dir}')
            return

        # unzip
        print_message(f'Unzipping {local_path}')
        unzipped_path = local_path.replace('.gz', '')
        with gzip.open(local_path, 'rb') as f_in, open(unzipped_path, 'wb') as f_out:
            f_out.write(f_in.read())

        # The published file is a pickled dictionary of float16 rows. Convert it into the store, which
        # compares every stored row with the file, and never embed what the file already holds.
        if not self._pooling_order_is_recorded():
            print_message(
                f'Not converting {filename}: it names its pooling types sorted, and this run pools in the '
                f'order {self.pooling_types}, so the published columns may not line up. '
                'Convert it by hand with `python -m protify.legacy_cache <file> --pooling <order>`.'
            )
            return None
        store_exists = has_feature(locate_feature(self.embedding_save_dir, descriptor))
        if store_exists:
            print_message(f'Found existing embeddings for {model_name} in {self.embedding_save_dir}, adding the downloaded sequences they lack')
        else:
            print_message(f'Downloading embeddings from {self.download_dir}, no previous embeddings found')
        # Existing rows are kept as they are, so a store that already has a sequence never takes the
        # published row. Rows widen to the store's dtype and are never narrowed.
        conversion = convert_legacy_cache(unzipped_path, self.embedding_save_dir, descriptor, skip_existing=store_exists)
        return None if conversion is None else str(conversion.directory)

    def _pooling_order_is_recorded(self) -> bool:
        """Whether an old cache's pooling order can be read from its name.

        A file name lists the pooling types sorted, while the Pooler concatenates them in the order it was
        given, so the name says which columns a file holds and not their order. It is safe to label an
        old cache with this run's order only when that order is the sorted one.
        """
        return self.matrix_embed or list(self.pooling_types) == sorted(self.pooling_types)

    def _adopt_legacy_caches(
            self,
            model_name: str,
            descriptor: dict[str, Any],
            model_type: str | None = None,
            model_path: str | None = None,
        ) -> None:
        """Convert this representation's old cache files into a store that does not exist yet.

        The file the run's own mode wrote comes first, so its rows win where both formats hold a
        sequence, and the other file only adds sequences the store lacks. A store that already exists is
        left alone: convert a file into it by hand with `python -m protify.legacy_cache --skip_existing`.
        """
        if has_feature(locate_feature(self.embedding_save_dir, descriptor)):
            return
        refusal = legacy_conversion_refusal(descriptor)
        if refusal is not None:
            print_message(refusal)
            return
        extensions = ('db', 'pth') if self.sql else ('pth', 'db')
        old_files = self.args.legacy_cache_files(model_name, None, model_type, model_path, extensions)
        if not old_files:
            return
        if not self._pooling_order_is_recorded():
            print_message(
                f'Not converting {old_files}: an old cache names its pooling types sorted, and this run pools '
                f'in the order {self.pooling_types}. Convert a file by hand with '
                '`python -m protify.legacy_cache <file> --pooling <order>` once the order is known.'
            )
            return
        for position, path in enumerate(old_files):
            print_message(f'Converting the old cache {path} into the feature store under {self.embedding_save_dir}')
            convert_legacy_cache(path, self.embedding_save_dir, descriptor, skip_existing=position > 0)

    def _read_embeddings_from_disk(
            self,
            model_name: str,
            model_type: str | None = None,
            model_path: str | None = None,
        ) -> tuple[list[str], str, dict[str, torch.Tensor]]:
        """Find what the store already holds: the sequences still to embed, the store, and any rows to keep.

        Returns `(to_embed, save_path, embeddings_dict)`. `save_path` is the store's directory, present or
        not yet made. Under `sql` the rows stay on disk and the dictionary is empty; otherwise it holds
        the already embedded rows of this run's sequences, read once.
        """
        descriptor = self.args.cache_descriptor(model_name, model_type=model_type, model_path=model_path)
        self._adopt_legacy_caches(model_name, descriptor, model_type, model_path)
        directory = locate_feature(self.embedding_save_dir, descriptor)
        sequences = list(dict.fromkeys(self.all_seqs))
        if not has_feature(directory):
            print_message(f"No embeddings found in {directory}")
            return sequences, str(directory), {}  # empty dict: no rows (d,) or (l + 2, d) cached yet

        to_embed = list(missing_sequences(directory, sequences))
        absent = set(to_embed)
        already_embedded = [sequence for sequence in sequences if sequence not in absent]
        print_message(f"Found {len(already_embedded)} already embedded sequences in {directory}\nEmbedding {len(to_embed)} new sequences")
        embeddings_dict = {} if self.sql else read_embeddings(directory, already_embedded)
        return to_embed, str(directory), embeddings_dict  # sequences still to embed, store directory, sequence -> (d,) pooled or (l + 2, d) token rows

    @torch.inference_mode()
    def _embed_sequences(
            self,
            to_embed: list[str],
            save_path: str,
            embedding_model: Any,
            tokenizer: Any,
            embeddings_dict: dict[str, torch.Tensor],
            descriptor: dict[str, Any] | None = None) -> dict[str, torch.Tensor] | None:
        """Embed `to_embed`, keeping the rows in `embeddings_dict` unless `sql` leaves them on disk.

        A `descriptor` (`EmbeddingArguments.cache_descriptor`) makes the run write its rows into the store
        as it goes, one committed window at a time, so an interrupted run resumes with only the sequences
        the store still lacks. Without one nothing is written. `save_path` is the store's directory.
        """
        # embeddings_dict: (d,) pooled or (l + 2, d) token rows by sequence; each row covers CLS, the l residues and EOS.
        os.makedirs(self.embedding_save_dir, exist_ok=True)
        model = embedding_model.to(self.device).eval()
        # SAE models reduce inside the encoder and return (b, p * c), so a Pooler here would
        # never be called, and would reject the names only they accept, such as 'sum'.
        pools_internally = getattr(model, "pools_internally", False)
        dynamic = self.padding == 'longest'
        model = maybe_compile(model, dynamic=dynamic)
        device = self.device
        collate_fn = build_collator(tokenizer, padding=self.padding, max_length=self.max_length)
        if self.matrix_embed or pools_internally:
            pooler = None
        else:
            print_message(f'Pooling types: {self.pooling_types}')
            pooler = Pooler(self.pooling_types)
        pooling_token_id = getattr(tokenizer, 'pooling_token_id', None)

        def _get_embeddings(
                residue_embeddings: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                attentions: torch.Tensor | None = None,
                input_ids: torch.Tensor | None = None,
            ) -> torch.Tensor:
            # residue_embeddings: (b, n, d) token rows, or (b, d) from a model that pools inside;
            # attention_mask: (b, n), 1 for CLS, residues and EOS and 0 for padding, so a mean covers all l + 2 rows;
            # attentions: (b, h, n, n); input_ids: (b, n).
            if residue_embeddings.ndim == 2 or self.matrix_embed:
                return residue_embeddings  # (b, n, d) per token or (b, d)
            else:
                return pooler(  # (b, d), pooled over the l + 2 rows the mask keeps
                    emb=residue_embeddings,
                    attention_mask=attention_mask,
                    attentions=attentions,
                    input_ids=input_ids,
                    eos_token_id=pooling_token_id,
                )

        to_embed = sorted(to_embed, key=len, reverse=True)
        dataset = SimpleProteinDataset(to_embed)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=dataloader_generator(get_global_seed())
        )

        # A run that raises keeps every window it committed and drops the open one.
        writer_context = (
            nullcontext() if descriptor is None
            else CacheWriter(self.embedding_save_dir, descriptor, layout=embedding_layout(descriptor))
        )
        with writer_context as writer:
            for i, batch in _make_embedding_progress(dataloader, self.padding):
                seqs = to_embed[i * self.batch_size:(i + 1) * self.batch_size]
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask']
                elif 'sequence_ids' in batch:
                    attention_mask = (batch['sequence_ids'] != -1).long().to(device)
                else:
                    attention_mask = torch.ones_like(batch['input_ids'], device=device)
                hidden_kwargs = {}
                if self.hidden_state_index != -1:
                    hidden_kwargs['output_hidden_states'] = True
                    hidden_kwargs['hidden_state_index'] = self.hidden_state_index

                with torch.autocast(device.type, dtype=self.embed_dtype, enabled=self.autocast):
                    if 'parti' in self.pooling_types:
                        # A failure here used to switch the whole run to mean pooling, which
                        # changed the embedding width partway through a cache and left the
                        # filename claiming 'parti'. Let it raise instead.
                        residue_embeddings, attentions = model(
                            **batch,
                            output_attentions=True,
                            **hidden_kwargs,
                        )
                        embeddings = _get_embeddings(
                            residue_embeddings,
                            attention_mask=attention_mask,
                            attentions=attentions,
                            input_ids=batch.get('input_ids'),
                        ).cpu()
                    else:
                        residue_embeddings = model(**batch, **hidden_kwargs)
                        embeddings = _get_embeddings(
                            residue_embeddings,
                            attention_mask=attention_mask,
                            input_ids=batch.get('input_ids'),
                        ).cpu()

                embeddings = embeddings.to(self.embed_dtype)  # (b, d) pooled, or (b, n, d) per token with n the longest l + 2 (CLS, l residues, EOS, then padding)
                if self.matrix_embed:
                    rows = [emb[mask.bool()] for emb, mask in zip(embeddings, attention_mask.cpu(), strict=True)]  # (l_i + 2, d) each: the mask keeps CLS, residues and EOS, drops padding
                else:
                    rows = list(embeddings)  # (d,) each
                if writer is not None:
                    writer.add(seqs, rows)
                if not self.sql:
                    embeddings_dict.update(zip(seqs, rows, strict=True))

        if writer is not None:
            print_message(f"Wrote {writer.rows_written} embeddings in {writer.windows} windows to {save_path}")
        return embeddings_dict  # sequence -> (d,) pooled or (l + 2, d) token rows

    def _embed_sequences_multi_gpu(
            self,
            to_embed: list[str],
            save_path: str,
            dispatch_name: str,
            model_path: str | None,
            embeddings_dict: dict[str, torch.Tensor],
            descriptor: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Embed `to_embed` with one worker per GPU, each on its own shard of the sequences.

        With a `descriptor`, `sql` workers commit their own windows to the store, which the store's
        locks allow, and otherwise the parent writes the merged rows once the workers finish.
        """
        # embeddings_dict: (d,) pooled or (l + 2, d) token rows by sequence, as in `_embed_sequences`.
        n_gpus = torch.cuda.device_count()
        to_embed = sorted(to_embed, key=len, reverse=True)
        chunk_size = math.ceil(len(to_embed) / n_gpus)
        shards = [to_embed[i * chunk_size:(i + 1) * chunk_size] for i in range(n_gpus)]
        shards = [s for s in shards if len(s) > 0]
        actual_gpus = len(shards)
        print_message(f"Multi-GPU: splitting {len(to_embed)} sequences across {actual_gpus} GPUs")

        if self.sql:
            os.makedirs(self.embedding_save_dir, exist_ok=True)

        result_dicts: dict[int, dict[str, torch.Tensor]] = mp.Manager().dict()

        def _worker(rank: int, shard: list[str]) -> None:
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
            model_obj, tokenizer = get_base_model(
                dispatch_name,
                dtype=self.model_dtype,
                model_path=model_path,
                pooling_types=None if self.matrix_embed else self.pooling_types,
            )
            model_obj = model_obj.to(device).eval()
            pools_internally = getattr(model_obj, "pools_internally", False)
            dynamic = self.padding == 'longest'
            model_obj = maybe_compile(model_obj, dynamic=dynamic)
            collate_fn = build_collator(tokenizer, padding=self.padding, max_length=self.max_length)

            if self.matrix_embed or pools_internally:
                pooler = None
            else:
                pooler = Pooler(self.pooling_types)
            pooling_token_id = getattr(tokenizer, 'pooling_token_id', None)

            def _get_embeddings(
                    residue_embeddings: torch.Tensor,
                    attention_mask: torch.Tensor | None=None,
                    attentions: tuple[torch.Tensor, ...] | None=None,
                    input_ids: torch.Tensor | None=None,
                ) -> torch.Tensor:
                # residue_embeddings: (b, n, d) token rows, or (b, d) from a model that pools inside, attention_mask: (b, n), 1 for CLS, residues and EOS and 0 for padding, attentions: (b, h, n, n) per layer, input_ids: (b, n)
                if residue_embeddings.ndim == 2 or self.matrix_embed:
                    return residue_embeddings  # (b, n, d) per token or (b, d)
                return pooler(  # (b, d), pooled over the l + 2 rows the mask keeps
                    emb=residue_embeddings,
                    attention_mask=attention_mask,
                    attentions=attentions,
                    input_ids=input_ids,
                    eos_token_id=pooling_token_id,
                )

            dataset = SimpleProteinDataset(shard)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=2 if self.num_workers > 0 else None,
                collate_fn=collate_fn,
                shuffle=False,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=dataloader_generator(get_global_seed()),
            )

            local_dict = {}
            worker_writer = (
                CacheWriter(self.embedding_save_dir, descriptor, layout=embedding_layout(descriptor), label=f'gpu{rank}')
                if self.sql else None
            )

            with torch.inference_mode():
                for i, batch_data in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    desc=f'GPU {rank}',
                    position=rank,
                ):
                    seqs = shard[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_data = {k: v.to(device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
                    if 'attention_mask' in batch_data:
                        attention_mask = batch_data['attention_mask']
                    elif 'sequence_ids' in batch_data:
                        attention_mask = (batch_data['sequence_ids'] != -1).long().to(device)
                    else:
                        attention_mask = torch.ones_like(batch_data['input_ids'], device=device)
                    hidden_kwargs = {}
                    if self.hidden_state_index != -1:
                        hidden_kwargs['output_hidden_states'] = True
                        hidden_kwargs['hidden_state_index'] = self.hidden_state_index

                    with torch.autocast(device.type, dtype=self.embed_dtype, enabled=self.autocast):
                        residue_embeddings = model_obj(**batch_data, **hidden_kwargs)
                        embeddings = _get_embeddings(
                            residue_embeddings,
                            attention_mask=attention_mask,
                            input_ids=batch_data.get('input_ids'),
                        ).cpu()

                    embeddings = embeddings.to(self.embed_dtype)  # (b, d) pooled, or (b, n, d) per token with n the longest l + 2 (CLS, l residues, EOS, then padding)
                    if self.matrix_embed:
                        rows = [emb[mask.bool()] for emb, mask in zip(embeddings, attention_mask.cpu(), strict=True)]  # (l_i + 2, d) each: the mask keeps CLS, residues and EOS, drops padding
                    else:
                        rows = list(embeddings)  # (d,) each
                    if worker_writer is not None:
                        worker_writer.add(seqs, rows)
                    else:
                        local_dict.update(zip(seqs, rows, strict=True))

            if worker_writer is not None:
                worker_writer.close()
            else:
                result_dicts[rank] = local_dict

        processes = []
        mp.set_start_method('spawn', force=True)
        for rank, shard in enumerate(shards):
            p = mp.Process(target=_worker, args=(rank, shard))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        if not self.sql:
            new_rows: dict[str, torch.Tensor] = {}
            for rank in range(actual_gpus):
                new_rows.update(result_dicts[rank])
            embeddings_dict.update(new_rows)
            if descriptor is not None:
                print_message(f"Saving embeddings to {save_path}")
                with CacheWriter(self.embedding_save_dir, descriptor, layout=embedding_layout(descriptor)) as writer:
                    writer.add(list(new_rows), list(new_rows.values()))

        return embeddings_dict  # sequence -> (d,) pooled or (l + 2, d) token rows

    def __call__(self, model_name: str, model_type: str = None, model_path: str = None) -> dict[str, torch.Tensor] | None:
        # A control is generated where it is read (feature_cache), so there is nothing to download or embed.
        if self.download_embeddings and not is_generated_model(model_name):
            self._download_embeddings(model_name, model_type=model_type, model_path=model_path)

        if self.device.type == 'cpu':
            warnings.warn("Downloading embeddings is recommended for CPU usage - Embedding on CPU will be extremely slow!", stacklevel=2)
        to_embed, save_path, embeddings_dict = self._read_embeddings_from_disk(
            model_name,
            model_type=model_type,
            model_path=model_path,
        )

        if len(to_embed) > 0:
            print_message(f"Embedding {len(to_embed)} sequences with {model_name}")
            dispatch_name = model_type or model_name
            # Rows are written to the store when they are streamed (sql) or when the run keeps them.
            descriptor = (
                self.args.cache_descriptor(model_name, model_type=model_type, model_path=model_path)
                if self.sql or self.save_embeddings else None
            )

            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if self.multi_gpu and n_gpus > 1:
                return self._embed_sequences_multi_gpu(  # sequence -> (d,) pooled or (l + 2, d) token rows
                    to_embed, save_path, dispatch_name, model_path, embeddings_dict, descriptor,
                )
            else:
                # SAE models fuse pooling into their encoder pass, so they need the
                # pooling choice up front and return pooled vectors directly.
                model, tokenizer = get_base_model(
                    dispatch_name,
                    dtype=self.model_dtype,
                    model_path=model_path,
                    pooling_types=None if self.matrix_embed else self.pooling_types,
                )
                return self._embed_sequences(to_embed, save_path, model, tokenizer, embeddings_dict, descriptor)  # sequence -> (d,) pooled or (l + 2, d) token rows
        else:
            print_message(f"No sequences to embed with {model_name}")
            return embeddings_dict  # sequence -> (d,) pooled or (l + 2, d) token rows


if __name__ == '__main__':
    ### Embed all supported datasets with all supported models
    # py -m embedder
    import argparse
    import tempfile
    from huggingface_hub import upload_file
    from .data.supported_datasets import vector_benchmark
    from .data.data_mixin import DataArguments, DataMixin
    from .base_models.get_base_models import BaseModelArguments, get_base_model
    from .seed_utils import set_global_seed

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--embed_dtype', type=str, default='float16')
    parser.add_argument('--model_names', nargs='+', default=['standard'])
    parser.add_argument('--models_to_skip', nargs='+', default=[], help='When checking for existing embeddings, skip these models.')
    parser.add_argument('--embedding_save_dir', type=str, default='embeddings')
    parser.add_argument('--download_dir', type=str, default='Synthyra/vector_embeddings')
    parser.add_argument('--embedding_pooling_types', nargs='+', default=['mean', 'var'], help='Pooling types for embeddings.')
    parser.add_argument('--embedding_hidden_state_index', type=int, default=-1, help='Hidden-state tuple index for embeddings. -1 uses the final hidden state.')
    args = parser.parse_args()

    chosen_seed = set_global_seed()

    if args.embed_dtype == 'float16':
        dtype = torch.float16
    elif args.embed_dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif args.embed_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid embedding dtype: {args.embed_dtype}")

    # Get data    
    data_args = DataArguments(
        data_names=vector_benchmark,
        max_length=1024,
        trim=False
    )
    all_seqs = DataMixin(data_args).get_data()[1]

    # Embed for each model
    model_args = BaseModelArguments(model_names=args.model_names)
    for model_name in model_args.model_names:

        embedder_args = EmbeddingArguments(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download_embeddings=model_name not in args.models_to_skip,
            matrix_embed=False,
            embedding_pooling_types=args.embedding_pooling_types,
            save_embeddings=True,
            embed_dtype=dtype,
            sql=False,
            embedding_save_dir=args.embedding_save_dir,
            embedding_hidden_state_index=args.embedding_hidden_state_index,
        )
        embedder = Embedder(embedder_args, all_seqs)

        _ = embedder(model_name)
        filename = get_embedding_filename(
            model_name,
            False,
            embedder_args.pooling_types,
            'pth',
            embedder_args.hidden_state_index,
            max_length=embedder_args.max_length,
        )
        # The download repository holds one gzipped pickled dictionary of rows per representation, the
        # file `_download_embeddings` converts. Export the rows from the store into it in a temporary
        # directory outside the tree, upload it, and keep nothing: the store is the only local copy.
        rows = read_embeddings(embedder_args.feature_directory(model_name), all_seqs)
        with tempfile.TemporaryDirectory() as staging:
            export_path = os.path.join(staging, filename)
            torch.save({sequence: row.to(dtype) for sequence, row in rows.items()}, export_path)
            compressed_path = f"{export_path}.gz"
            print(f"Compressing {export_path} to {compressed_path}")
            with open(export_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
            upload_file(
                path_or_fileobj=compressed_path,
                path_in_repo=f'embeddings/{filename}.gz',
                repo_id=args.download_dir,
                repo_type='dataset'
            )

    print('Done')
