"""Offline allowlist for the exact course-approved BERT snapshot."""

import hashlib
from pathlib import Path

MODEL = "google-bert/bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
SOURCE_URL = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
# Verified against the official revision API on 2026-11-14. For small files,
# the downloaded bytes were matched to the API's Git blob IDs before pinning SHA-256.
# This allowlist is code-owned; never regenerate it from an arbitrary local directory.
APPROVED_FILES = {
    "config.json": (570, "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20"),
    "model.safetensors": (440449768, "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"),
    "tokenizer.json": (466062, "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"),
    "tokenizer_config.json": (48, "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"),
    "vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_model(directory):
    """Reject unapproved bytes before loading models or creating run artifacts."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Base model directory does not exist: {directory}; run main.py prepare")
    # Extra tokenizer/adapter/config files can alter from_pretrained behavior even
    # when all five required files match. Only the unused download cache is allowed.
    extras = [p.name for p in directory.iterdir()
              if p.name not in APPROVED_FILES and not (p.name == ".cache" and p.is_dir())]
    if extras:
        raise ValueError(f"Unapproved files in base model directory: {sorted(extras)}; "
                         "use main.py prepare --base-model with a new empty directory")
    verified = {}
    for name, (size, expected) in APPROVED_FILES.items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing approved base model file: {name}")
        if path.stat().st_size != size:
            raise ValueError(f"Base model size mismatch: {name}; expected the pinned {MODEL} snapshot")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Base model SHA-256 mismatch: {name}; expected the pinned {MODEL} snapshot")
        verified[name] = {"bytes": size, "sha256": actual}
    return {"model": MODEL, "revision": REVISION, "directory": str(directory),
            "verification": "pinned-files-v1", "reference": SOURCE_URL, "files": verified}
