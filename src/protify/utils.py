import io
import os
import struct
import torch
import shutil
import pyfiglet
from functools import partial
from typing import List, Optional, Tuple

import numpy as np

torch_load = partial(torch.load, map_location='cpu', weights_only=True)

_COMPACT_VERSION = 0x01
_DTYPE_TO_CODE = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_CODE_TO_DTYPE = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32}
_CODE_TO_NP_DTYPE = {0: np.float16, 1: np.float16, 2: np.float32}


def tensor_to_embedding_blob(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to compact binary format for SQLite blob storage.

    Format: [version:1][dtype_code:1][ndim:4][shape:4*ndim][raw_bytes]
    bfloat16 tensors are stored as float16 bytes (numpy lacks bfloat16)
    but tagged with dtype_code=1 so they can be cast back on read.
    Falls back to torch.save for unsupported dtypes.
    """
    t = tensor.cpu()
    if t.dtype not in _DTYPE_TO_CODE:
        buffer = io.BytesIO()
        torch.save(t, buffer)
        return buffer.getvalue()
    dtype_code = _DTYPE_TO_CODE[t.dtype]

    if t.dtype == torch.bfloat16:
        raw = t.half().numpy().tobytes()
    else:
        raw = t.numpy().tobytes()

    shape = t.shape
    header = struct.pack(f'<BBi{len(shape)}i', _COMPACT_VERSION, dtype_code, len(shape), *shape)
    return header + raw


def _compact_header(dtype: torch.dtype, shape: tuple) -> bytes:
    """Build just the compact header for a given dtype and shape."""
    dtype_code = _DTYPE_TO_CODE[dtype]
    return struct.pack(f'<BBi{len(shape)}i', _COMPACT_VERSION, dtype_code, len(shape), *shape)


def batch_tensor_to_blobs(batch: torch.Tensor) -> list:
    """Serialize a batch of identically-shaped embeddings to compact blobs.

    Input: (B, D) or (B, L, D) tensor already on CPU and in target dtype.
    Returns: list of B bytes objects, one per embedding.
    Much faster than calling tensor_to_embedding_blob() per row because
    dtype cast, numpy conversion, and header construction happen once.
    """
    assert batch.dtype in _DTYPE_TO_CODE, f"Unsupported dtype {batch.dtype}"
    single_shape = tuple(batch.shape[1:])
    header = _compact_header(batch.dtype, single_shape)

    if batch.dtype == torch.bfloat16:
        raw = batch.half().numpy().tobytes()
    else:
        raw = batch.numpy().tobytes()

    stride = raw.__len__() // batch.shape[0]
    return [header + raw[i * stride:(i + 1) * stride] for i in range(batch.shape[0])]


def embedding_blob_to_tensor(
    blob: bytes,
    fallback_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Deserialize an embedding blob from SQLite.

    Tries compact binary format first (version byte 0x01), then PyTorch
    torch.save format, then legacy raw float32 with fallback_shape.
    """
    if len(blob) >= 2 and blob[0] == _COMPACT_VERSION and blob[1] in _CODE_TO_DTYPE:
        dtype_code = blob[1]
        ndim = struct.unpack_from('<i', blob, 2)[0]
        shape = struct.unpack_from(f'<{ndim}i', blob, 6)
        data_offset = 6 + 4 * ndim
        np_dtype = _CODE_TO_NP_DTYPE[dtype_code]
        arr = np.frombuffer(blob, dtype=np_dtype, offset=data_offset).reshape(shape).copy()
        t = torch.from_numpy(arr)
        if dtype_code == 1:
            t = t.to(torch.bfloat16)
        return t

    try:
        t = torch_load(io.BytesIO(blob))
        if isinstance(t, torch.Tensor):
            return t
    except Exception:
        pass
    if fallback_shape is not None:
        return torch.tensor(
            np.frombuffer(blob, dtype=np.float32).reshape(fallback_shape)
        )
    raise ValueError(
        "Blob is not in compact/PyTorch format and no fallback_shape provided for legacy float32."
    )


def clear_screen() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def print_message(message: str) -> None:
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 50
    print('\n' + '-' * terminal_width)
    print(f'\n{message}\n')
    print('-' * terminal_width + '\n')


def print_title(title: str) -> None:
    print(pyfiglet.figlet_format(title, font='3d-ascii'))


def print_done() -> None:
    print(pyfiglet.figlet_format('== Done ==', font='js_stick_letters'))


def expand_dms_ids_all(dms_ids: List[str], mode: Optional[str] = None) -> List[str]:
    """
    Expand 'all' to actual DMS IDs from benchmarks.proteingym.dms_ids.
    """
    if any(str(x).lower() == 'all' for x in dms_ids):
        if mode == 'indels':
            from benchmarks.proteingym.dms_ids import ALL_INDEL_DMS_IDS
            dms_ids = list(ALL_INDEL_DMS_IDS)
        else:
            from benchmarks.proteingym.dms_ids import ALL_SUBSTITUTION_DMS_IDS
            dms_ids = list(ALL_SUBSTITUTION_DMS_IDS)
    return dms_ids


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
    except Exception as e:
        print_message(f"Skipping torch.compile: {e}")
    return model


if __name__ == '__main__':
    folders_to_clean = ['logs', 'results', 'plots', 'embeddings', 'weights']
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            files = os.listdir(folder)
            if files:
                response = input(f"Do you want to delete all files in '{folder}' folder? ({len(files)} files) [y/N]: ")
                if response.lower() == 'y':
                    for file in files:
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    print(f"All files in '{folder}' have been deleted.")
                else:
                    print(f"Skipped cleaning '{folder}' folder.")
            else:
                print(f"'{folder}' folder is already empty.")
        else:
            print(f"'{folder}' folder does not exist.")
