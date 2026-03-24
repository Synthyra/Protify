import io
import os
import torch
import shutil
import pyfiglet
from functools import partial
from typing import List, Optional, Tuple

import numpy as np

torch_load = partial(torch.load, map_location='cpu', weights_only=True)


def tensor_to_embedding_blob(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes for SQLite blob storage. Supports all torch dtypes."""
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return buffer.getvalue()


def embedding_blob_to_tensor(
    blob: bytes,
    fallback_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Deserialize an embedding blob from SQLite. Tries PyTorch format first; on failure
    treats blob as legacy raw float32 and reshapes with fallback_shape.
    """
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
        "Blob is not PyTorch-serialized and no fallback_shape provided for legacy float32."
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
