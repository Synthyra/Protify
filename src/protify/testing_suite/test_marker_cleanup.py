from pathlib import Path


def test_first_party_marker_cleanup() -> None:
    root = Path(__file__).resolve().parents[1]
    markers = ("TO" + "DO", "FIX" + "ME", "X" * 3)
    text_suffixes = {".py", ".yaml", ".yml", ".md", ".txt"}
    offenders = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "fastplms" in path.parts:
            continue
        if path.suffix.lower() == ".ipynb":
            continue
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker in text:
                offenders.append(f"{path.relative_to(root)} contains {marker}")

    assert offenders == []
