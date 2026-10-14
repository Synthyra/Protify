"""Restartable pipeline stages that are either complete on disk or absent.

A dataset build runs for hours, and a crash halfway through a stage must not leave outputs a
rerun mistakes for finished ones. Each stage is built in a temporary sibling directory and
renamed into place only once every output exists, with a marker written last. Promoted from
latent_sep, whose MMseqs database, clustering, and audit stages use it.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from collections.abc import Callable, Sequence
from pathlib import Path


COMPLETE_MARKER = ".complete.json"


def run_stage(directory: Path, outputs: Sequence[str], build: Callable[[Path], None]) -> Path:
    """Build `outputs` into `directory` once, and return `directory`.

    `build` receives the temporary directory to write into. A `directory` already marked
    complete is returned as it is, so a rerun resumes after it. An unmarked one is refused: it
    is a crashed attempt, and neither reusing nor deleting it is safe to assume.
    """
    marker = directory / COMPLETE_MARKER
    if marker.is_file() and all(output_exists(directory / output) for output in outputs):
        return directory
    if directory.exists():
        raise RuntimeError(
            f"Incomplete stage directory exists; move or remove it before retrying: {directory}"
        )

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        build(temporary)
        missing = [output for output in outputs if not output_exists(temporary / output)]
        if missing:
            raise RuntimeError(f"Stage did not create: {', '.join(missing)}")
        (temporary / COMPLETE_MARKER).write_text(
            json.dumps({"required_outputs": list(outputs)}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory


def output_exists(path: Path) -> bool:
    """A regular output, or a multipart MMseqs database such as `sequence_db.index`."""
    return path.exists() or next(path.parent.glob(f"{path.name}.*"), None) is not None
