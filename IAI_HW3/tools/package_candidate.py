"""Build a whitelist-only rehearsal ZIP for the registered single-model candidate."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from tools.verify_package import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ROOT / "outputs") or destination.exists():
        raise ValueError("Use a new build directory under HW3 outputs")
    registry = json.loads((ROOT / "experiments/registry.json").read_text(encoding="utf-8"))
    snapshot = json.loads((ROOT / "experiments/snapshot.json").read_text(encoding="utf-8"))
    candidate = registry["kaggle_candidate"]
    entry = next(r for r in registry["runs"] if r["id"] == candidate)
    run = next(r for r in snapshot["runs"] if r["id"] == candidate)
    if entry["kind"] != "single" or len(entry["checkpoints"]) != 1:
        raise ValueError("Rehearsal builder requires a single-model candidate")
    checkpoint = entry["checkpoints"][0]
    expected = snapshot["checkpoints"][checkpoint]["files_sha256"]
    for name, checksum in expected.items():
        if digest(ROOT / checkpoint / name) != checksum:
            raise ValueError(f"Candidate checkpoint changed: {name}")
    if digest(ROOT / run["predictions"]) != run["predictions_sha256"]:
        raise ValueError("Candidate CSV changed")
    for name, key in (("dataset/dataset.csv", "dataset_sha256"),
                      ("dataset/benchmark.csv", "benchmark_sha256"),
                      ("outputs/eda/split.json", "split_sha256")):
        if digest(ROOT / name) != snapshot[key]:
            raise ValueError(f"Source changed: {name}")
    sources = {name: name for name in (
        "main.py", "requirements.txt", "requirements/train.txt", "requirements/eda.txt", "requirements/lock.txt",
        "src/__init__.py", "src/training.py", "src/eda.py", "src/preprocessing.py", "src/model_source.py",
        "dataset/dataset.csv", "dataset/benchmark.csv", "outputs/eda/split.json")}
    sources.update({"README.md": "docs/Package_README.md", "verify_package.py": "tools/verify_package.py",
                    "reference/predictions.csv": run["predictions"]})
    sources.update({f"saved_models/checkpoint/{name}": f"{checkpoint}/{name}" for name in expected})
    sources.update({f"training_artifacts/{name}": f"{entry['output']}/{name}" for name in
                    ("curves.png", "history.json", "run_config.json", "summary.json", "environment.json")})
    package_name = f"HW3_rehearsal_{candidate}"
    package = destination / package_name
    package.mkdir(parents=True)
    manifest = {"schema_version": 1, "purpose": "rehearsal; not final submission", "candidate": candidate,
                "student_id": None, "release_candidate": registry["release_candidate"],
                "source_checkpoint": checkpoint, "reference_predictions_sha256": run["predictions_sha256"],
                "builder_sha256": digest(Path(__file__)), "files": {}}
    for relative, source in sources.items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
        checksum = digest(target)
        if checksum != digest(ROOT / source):
            raise ValueError(f"Copy verification failed: {relative}")
        manifest["files"][relative] = {"source": source, "bytes": target.stat().st_size, "sha256": checksum}
    (package / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify(package)
    archive = destination / f"{package_name}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                stream.write(path, f"{package_name}/{path.relative_to(package).as_posix()}")
    extracted = destination / "extracted"
    # Only extract the just-created whitelist archive into a new directory.
    with zipfile.ZipFile(archive) as stream:
        expected_names = {f"{package_name}/{p}" for p in (*sources, "package_manifest.json")}
        if set(stream.namelist()) != expected_names or len(stream.namelist()) != len(expected_names):
            raise ValueError("Unexpected ZIP content")
        stream.extractall(extracted)
    verified = verify(extracted / package_name)
    summary = {"candidate": candidate, "archive": str(archive.relative_to(ROOT).as_posix()),
               "archive_bytes": archive.stat().st_size, "archive_sha256": digest(archive),
               "extracted": str((extracted / package_name).relative_to(ROOT).as_posix()),
               "manifest_sha256": digest(package / "package_manifest.json"), **verified,
               "clean_environment_verified": False, "prediction_verified": False}
    (destination / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
