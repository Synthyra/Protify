import ctypes
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

# Raw binary serialization format for embedding blobs.
# Layout: magic (4 bytes "RAWT") + dtype_id (1 byte) + ndim (1 byte) + shape (4 bytes per dim) + raw tensor bytes
# torch.save blobs start with "PK\x03\x04" (zip) or "\x80\x02"/"\x80\x04" (pickle), so "RAWT" is unambiguous.
_BLOB_MAGIC = b'RAWT'
_DTYPE_TO_ID = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2, torch.float64: 3}
_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}


def tensor_to_embedding_blob(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes for SQLite blob storage.

    Uses a compact raw binary format: a 4-byte magic, a short header (dtype + shape),
    then a direct memcpy of the tensor's data buffer. This is ~10-50x faster than
    torch.save/pickle for large tensors and supports all dtypes (float32, float16,
    bfloat16, float64) natively.
    """
    t = tensor.cpu().contiguous()
    assert t.dtype in _DTYPE_TO_ID, f"Unsupported dtype: {t.dtype}"
    header = _BLOB_MAGIC + struct.pack(f'<BB{len(t.shape)}I', _DTYPE_TO_ID[t.dtype], len(t.shape), *t.shape)
    nbytes = t.nelement() * t.element_size()
    raw = (ctypes.c_char * nbytes).from_address(t.data_ptr())
    return header + bytes(raw)


def embedding_blob_to_tensor(
    blob: bytes,
    fallback_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Deserialize an embedding blob from SQLite.

    Tries the raw binary format first (RAWT magic prefix). Falls back to
    torch.load for legacy pickle-format blobs, then to raw float32 with
    fallback_shape for oldest legacy format.
    """
    # New raw binary format: starts with "RAWT"
    if len(blob) > 6 and blob[:4] == _BLOB_MAGIC:
        dtype_id, ndim = struct.unpack('<BB', blob[4:6])
        data_start = 6 + ndim * 4
        shape = struct.unpack(f'<{ndim}I', blob[6:data_start])
        dtype = _ID_TO_DTYPE[dtype_id]
        t = torch.empty(shape, dtype=dtype)
        ctypes.memmove(t.data_ptr(), blob[data_start:], len(blob) - data_start)
        return t

    # Legacy: torch.save pickle format
    try:
        t = torch_load(io.BytesIO(blob))
        if isinstance(t, torch.Tensor):
            return t
    except Exception:
        pass

    # Oldest legacy: raw float32 bytes
    if fallback_shape is not None:
        return torch.tensor(
            np.frombuffer(blob, dtype=np.float32).reshape(fallback_shape)
        )
    raise ValueError(
        "Blob is not raw-binary or PyTorch-serialized and no fallback_shape provided."
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


def maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    if os.name == 'posix':
        model = torch.compile(model, dynamic=False)
        print_message("Model compiled")
    else:
        print_message("Skipping torch.compile (not POSIX)")
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
