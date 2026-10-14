import importlib
import warnings
import torch

from typing import Any
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from .get_probe import rebuild_probe_from_saved_config
from ..base_models.sae_pooling import pool_sae_outputs
from ..base_models.supported_models import all_presets_with_paths
from ..data.pair_features import pair_vector_halves
from ..pooler import Pooler
from ..version import PROTIFY_VERSION


class PackagedProbeConfig(PretrainedConfig):
    model_type = "packaged_probe"

    def __init__(
        self,
        base_model_name: str = "",
        probe_type: str = "mlp",
        probe_config: dict[str, Any] | None = None,
        tokenwise: bool = False,
        matrix_embed: bool = False,
        pooling_types: list[str] | None = None,
        task_type: str = "singlelabel",
        num_labels: int = 2,
        ppi: bool = False,
        pair_feature_mode: str = "concat",
        add_token_ids: bool = False,
        sep_token_id: int | None = None,
        sae_repository: str | None = None,
        sae_revision: str | None = None,
        sae_layer: int | None = None,
        sae_codebook_dim: int | None = None,
        sae_k: int | None = None,
        feature_indices: list[int] | None = None,
        feature_input_size: int | None = None,
        standardized_width: int | None = None,
        protify_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Transformers 5 saves the label count only as `id2label`, which `super().__init__` restores
        # from a saved config. The default here would otherwise overwrite it on every load.
        if kwargs.get("id2label") and num_labels == 2:
            num_labels = len(kwargs["id2label"])
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.probe_type = probe_type
        self.probe_config = {} if probe_config is None else probe_config
        self.tokenwise = tokenwise
        self.matrix_embed = matrix_embed
        self.pooling_types = ["mean"] if pooling_types is None else pooling_types
        self.task_type = task_type
        self.num_labels = num_labels
        self.ppi = ppi
        # How the two pooled protein vectors are joined into the row the probe reads.
        # A probe trained on [min, max] cannot read [A, B], so this travels with it.
        self.pair_feature_mode = pair_feature_mode
        self.add_token_ids = add_token_ids
        self.sep_token_id = sep_token_id
        # A sparse autoencoder replaces dense pooling: the backbone emits codebook
        # features and `pooling_types` reduces those instead of the hidden states.
        self.sae_repository = sae_repository
        # The commit the codebook loads at; None in a repository exported before it was recorded.
        self.sae_revision = sae_revision
        self.sae_layer = sae_layer
        # The codebook's shape contract. Needed to rebuild the layer's structure when the
        # weights come from this checkpoint rather than from a fresh Hub download.
        self.sae_codebook_dim = sae_codebook_dim
        self.sae_k = sae_k
        # The columns the probe was trained on, and the width they were selected from.
        # Both are needed: applying an index array to the wrong width silently reindexes.
        self.feature_indices = feature_indices
        self.feature_input_size = feature_input_size
        # The width of the pooled row the probe's training standardizer was fit on, after any
        # feature selection; None when the probe read raw embeddings. Repositories exported
        # before this field existed carry no standardizer and keep reading raw embeddings.
        self.standardized_width = standardized_width
        # The Protify that trained and exported the probe; None for repositories that predate it.
        self.protify_version = protify_version


class PackagedProbeModel(PreTrainedModel):
    config_class = PackagedProbeConfig
    base_model_prefix = "backbone"
    all_tied_weights_keys = {}

    def __init__(
        self,
        config: PackagedProbeConfig,
        base_model: nn.Module | None = None,
        probe: nn.Module | None = None,
        standardizer: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.backbone = self._load_base_model() if base_model is None else base_model
        self.probe = self._load_probe() if probe is None else probe
        self.pooler = Pooler(self.config.pooling_types)
        if self.config.standardized_width is not None:
            # Persistent buffers, so the checkpoint carries the training split's statistics and
            # from_pretrained fills them; the placeholders below are overwritten on load.
            width = self.config.standardized_width  # s
            mean, scale = standardizer if standardizer is not None else (torch.zeros(width), torch.ones(width))
            assert mean.shape == (width,) and scale.shape == (width,), "standardizer width must match standardized_width"
            self.register_buffer("feature_mean", mean.detach().float().clone())  # (s,)
            self.register_buffer("feature_scale", scale.detach().float().clone())  # (s,)

    def _load_base_model(self) -> nn.Module:
        model_path = all_presets_with_paths.get(self.config.base_model_name, self.config.base_model_name)

        # Transformers 5 initializes custom architectures on the meta device
        # before loading the outer checkpoint. A nested from_pretrained call is
        # rejected in that context, and the packaged checkpoint already contains
        # the backbone weights, so build only the backbone structure on meta.
        if torch.get_default_device().type == "meta":
            backbone_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_config(backbone_config, trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        if self.config.sae_repository is not None:
            if torch.get_default_device().type == "meta":
                # load_sae_models places the codebook on the model's device, and
                # safetensors rejects meta, which is where from_pretrained builds. This
                # checkpoint carries the codebook weights, so only the structure is needed
                # here and the outer load fills it, exactly as it fills the backbone.
                model.add_sae_models([self._build_sae_structure(model.config.hidden_size)])
            else:
                model.load_sae_models(self.config.sae_repository, [self.config.sae_layer], revision=self.config.sae_revision)
        model.eval()
        return model

    def _build_sae_structure(self, d_model: int) -> nn.Module:
        """An unpopulated SAE layer of the recorded shape.

        Resolved through importlib rather than an import statement. FastPLMs is not a pip
        dependency of an exported repository: the backbone's own remote code installs it a
        few lines above this. Transformers scans a trust_remote_code file for import
        statements and refuses to load it when one names a package pip cannot find, so a
        plain `from fastplms ...` here would make every exported SAE repository unloadable.
        """
        sae_module = importlib.import_module(
            "fastplms.models.esm_plusplus.modeling_esm_plusplus_sae"
        )
        sae_layer_class = sae_module.ESMplusplusSAELayer
        sae_params_class = sae_module.ESMplusplusSAEParams

        missing = [
            name
            for name in ("sae_codebook_dim", "sae_k", "sae_layer")
            if getattr(self.config, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"This repository records a sparse autoencoder but not its shape: {missing}. "
                "Re-export it with a Protify version that persists the codebook contract."
            )
        return sae_layer_class(
            sae_params_class(
                d_model=d_model,
                codebook_dim=self.config.sae_codebook_dim,
                k=self.config.sae_k,
                layer=self.config.sae_layer,
            )
        )

    @property
    def uses_sae(self) -> bool:
        return self.config.sae_repository is not None

    def _select_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Keep the columns the probe was trained on.

        A probe trained on a selected subspace cannot read a full-width embedding, and an
        index array applied to the wrong width would reindex silently rather than fail,
        so the width is checked against the one selection was fit on.
        """
        # embeddings: (b, f) -> (b, s)
        if self.config.feature_indices is None:
            return embeddings  # (b, f) or (b, s)
        if embeddings.shape[-1] != self.config.feature_input_size:
            raise ValueError(
                f"Feature selection was fit on width {self.config.feature_input_size}, but "
                f"this embedding is {embeddings.shape[-1]} wide."
            )
        # Built per call from the config rather than held as a buffer. from_pretrained
        # constructs the module on the meta device and materializes only the keys the
        # checkpoint carries, so a buffer would still be meta here and index_select would
        # read out of bounds instead of raising.
        index = torch.as_tensor(
            self.config.feature_indices, dtype=torch.long, device=embeddings.device
        )  # (s,)
        return embeddings.index_select(-1, index)  # (b, s)

    def _standardize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Scale pooled rows as training did: the training split's mean and standard deviation, after selection."""
        # embeddings: (b, s)
        if self.config.standardized_width is None:
            return embeddings  # (b, s)
        if embeddings.shape[-1] != self.config.standardized_width:
            raise ValueError(
                f"The probe's standardizer was fit on width {self.config.standardized_width}, but "
                f"this embedding is {embeddings.shape[-1]} wide."
            )
        standardized = (embeddings.float() - self.feature_mean) / self.feature_scale  # (b, s)
        return standardized.to(embeddings.dtype)  # (b, s)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        """Refresh the backbone's non-persistent buffers after the checkpoint load.

        ESM++ registers its rotary tables with persistent=False, so they appear in no
        checkpoint and its own from_pretrained recomputes them afterwards. This model
        builds its backbone with from_config and then loads through its own checkpoint,
        which never runs that hook, leaving the tables as whatever the meta materialization
        produced. The symptom is a model whose predictions change from run to run.
        """
        loaded = super().from_pretrained(*args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        trained_with = model.config.protify_version
        if trained_with is not None and trained_with != PROTIFY_VERSION:
            warnings.warn(
                f"This probe was exported by Protify {trained_with} and is running on Protify {PROTIFY_VERSION}; "
                f"install protify=={trained_with} to reproduce its training-time predictions exactly.",
                stacklevel=2,
            )
        refresh = getattr(model.backbone, "_reset_rotary_embeddings", None)
        if callable(refresh):
            refresh()
        return loaded

    def _load_probe(self) -> nn.Module:
        return rebuild_probe_from_saved_config(
            probe_type=self.config.probe_type,
            tokenwise=self.config.tokenwise,
            probe_config=self.config.probe_config,
        )

    @staticmethod
    def _extract_hidden_states(backbone_output: Any) -> torch.Tensor:
        if isinstance(backbone_output, tuple):
            return backbone_output[0]  # (b, l, d)
        if hasattr(backbone_output, "last_hidden_state"):
            return backbone_output.last_hidden_state  # (b, l, d)
        if isinstance(backbone_output, torch.Tensor):
            return backbone_output  # (b, l, d)
        raise ValueError("Unsupported backbone output format for packaged probe model")

    @staticmethod
    def _extract_attentions(backbone_output: Any) -> Any | None:
        if hasattr(backbone_output, "attentions"):
            # Usually one (b, h, l, l) tensor per backbone layer.
            return backbone_output.attentions
        return None

    def _build_ppi_segment_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # input_ids: (b, l), attention_mask: (b, l), token_type_ids: (b, l) or None
        if token_type_ids is not None and torch.any(token_type_ids == 1):
            mask_a = ((token_type_ids == 0) & (attention_mask == 1)).long()  # (b, l)
            mask_b = ((token_type_ids == 1) & (attention_mask == 1)).long()  # (b, l)
            assert torch.all(mask_a.sum(dim=1) > 0), "PPI token_type_ids produced empty segment A"
            assert torch.all(mask_b.sum(dim=1) > 0), "PPI token_type_ids produced empty segment B"
            return mask_a, mask_b  # each (b, l)

        assert self.config.sep_token_id is not None, "sep_token_id is required for PPI fallback segmentation"
        batch_size, seq_len = input_ids.shape
        mask_a = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)  # (b, l)
        mask_b = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)  # (b, l)

        for batch_idx in range(batch_size):
            valid_positions = torch.where(attention_mask[batch_idx] == 1)[0]  # (n_valid,)
            sep_positions = torch.where(
                (input_ids[batch_idx] == self.config.sep_token_id)
                & (attention_mask[batch_idx] == 1)
            )[0]  # (n_sep,)
            if len(valid_positions) == 0:
                continue

            # n_a and n_b are the row-local token counts assigned to each segment.
            if len(sep_positions) >= 2:
                first_sep = int(sep_positions[0].item())
                second_sep = int(sep_positions[1].item())
                mask_a[batch_idx, :first_sep + 1] = 1  # (n_a,)
                mask_b[batch_idx, first_sep + 1:second_sep + 1] = 1  # (n_b,)
            elif len(sep_positions) == 1:
                first_sep = int(sep_positions[0].item())
                mask_a[batch_idx, :first_sep + 1] = 1  # (n_a,)
                mask_b[
                    batch_idx,
                    first_sep + 1: int(valid_positions[-1].item()) + 1,
                ] = 1  # (n_b,)
            else:
                midpoint = len(valid_positions) // 2
                mask_a[batch_idx, valid_positions[:midpoint]] = 1  # (n_a,)
                mask_b[batch_idx, valid_positions[midpoint:]] = 1  # (n_b,)

        assert torch.all(mask_a.sum(dim=1) > 0), "PPI fallback segmentation produced empty segment A"
        assert torch.all(mask_b.sum(dim=1) > 0), "PPI fallback segmentation produced empty segment B"
        return mask_a, mask_b  # each (b, l)

    def _build_probe_inputs(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        attentions: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: (b, l, d), input_ids: (b, l), attention_mask: (b, l), token_type_ids: (b, l) or None
        if self.config.ppi and (not self.config.matrix_embed) and (not self.config.tokenwise):
            mask_a, mask_b = self._build_ppi_segment_masks(
                input_ids,
                attention_mask,
                token_type_ids,
            )  # each (b, l)
            # p is the configured number of pooling operations.
            vec_a = self.pooler(hidden_states, attention_mask=mask_a, attentions=attentions)  # (b, p * d)
            vec_b = self.pooler(hidden_states, attention_mask=mask_b, attentions=attentions)  # (b, p * d)
            half_a, half_b = pair_vector_halves(
                vec_a, vec_b, self.config.pair_feature_mode
            )  # two of (b, p * d)
            probe_embeddings = torch.cat((half_a, half_b), dim=-1)  # (b, 2 * p * d)
            return probe_embeddings, None  # (b, 2 * p * d), None

        if self.config.matrix_embed or self.config.tokenwise:
            return hidden_states, attention_mask  # (b, l, d), (b, l)

        pooled = self.pooler(
            hidden_states,
            attention_mask=attention_mask,
            attentions=attentions,
        )  # (b, p * d)
        return pooled, None  # (b, p * d), None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput | TokenClassifierOutput:
        # input_ids: (b, l), attention_mask: (b, l) or None, token_type_ids: (b, l) or None, labels: (b,), (b, c), (b, l) or (b, l, c) by the task and probe.
        # Label shape is task- and probe-dependent.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)  # (b, l)

        requires_attentions = (
            "parti" in self.config.pooling_types
            and not self.config.matrix_embed
            and not self.config.tokenwise
        )
        backbone_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if requires_attentions:
            backbone_kwargs["output_attentions"] = True
        if self.uses_sae:
            backbone_kwargs["compute_sae"] = True
        backbone_output = self.backbone(**backbone_kwargs)

        if self.uses_sae:
            # The backbone emits codebook features over the batch's valid tokens instead
            # of hidden states, so pooling reduces those and the dense Pooler is unused.
            features = backbone_output.sae_outputs[f"layer{self.config.sae_layer}"].coalesce()
            probe_embeddings = pool_sae_outputs(
                features, attention_mask, tuple(self.config.pooling_types)
            )  # (b, p * c)
            probe_attention_mask = None
        else:
            hidden_states = self._extract_hidden_states(backbone_output)  # (b, l, d)
            attentions = self._extract_attentions(backbone_output)
            # Backbone-specific payload, typically per-layer (b, h, l, l) tensors, or None.
            if requires_attentions:
                assert attentions is not None, "parti pooling requires base model attentions"
            probe_embeddings, probe_attention_mask = self._build_probe_inputs(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                attentions=attentions,
            )  # (b, d_probe) or (b, l, d); mask is (b, l) or None

        probe_embeddings = self._select_features(probe_embeddings)
        if probe_attention_mask is None:
            # A pooled row; per-token inputs were never standardized.
            probe_embeddings = self._standardize(probe_embeddings)

        # Exported repositories predating the rename still carry "linear".
        if self.config.probe_type in ("mlp", "linear"):
            probe_output = self.probe(embeddings=probe_embeddings, labels=labels)
            return probe_output  # logits: (b, c)

        if self.config.probe_type == "transformer":
            forward_kwargs: dict[str, Any] = {
                "embeddings": probe_embeddings,
                "labels": labels,
            }
            if probe_attention_mask is not None:
                forward_kwargs["attention_mask"] = probe_attention_mask
            if self.config.add_token_ids and token_type_ids is not None and probe_attention_mask is not None:
                forward_kwargs["token_type_ids"] = token_type_ids
            probe_output = self.probe(**forward_kwargs)
            return probe_output  # logits: (b, c) or (b, l, c)

        probe_output = self.probe(
            embeddings=probe_embeddings,
            attention_mask=probe_attention_mask,
            labels=labels,
        )
        return probe_output  # External boundary; logits rank follows the registered probe.
