#!/usr/bin/env python3
"""Download and validate the evaluation corpora."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpora import SUPPORTED_CORPORA, validate_emphassess_directory


HF_DATASETS = {
    "tinystress": "slprl/TinyStress-15K",
    "stresstest": "slprl/StressTest",
    "stresspresso": "slprl/StressPresso",
    "expresso": "ylacombe/expresso",
}
HF_DATASET_CONFIGS = {"expresso": "read"}
EMPHASSESS_URL = "https://dl.fbaipublicfiles.com/speech_expressivity_evaluation/EmphAssess/EmphAssess_Dataset.tar.gz"


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"Unsafe path in archive: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not allowed in archive: {member.name}")
    archive.extractall(destination)


def download_hf_corpus(corpus: str, data_root: Path, force: bool) -> None:
    from datasets import load_dataset, load_from_disk

    destination = data_root / corpus
    if destination.exists() and not force:
        load_from_disk(str(destination))
        print(f"{corpus}: existing validated download; skipping")
        return
    if force:
        shutil.rmtree(destination, ignore_errors=True)
    load_kwargs = {}
    if corpus in HF_DATASET_CONFIGS:
        load_kwargs["name"] = HF_DATASET_CONFIGS[corpus]
    dataset = load_dataset(HF_DATASETS[corpus], **load_kwargs)
    temporary = Path(tempfile.mkdtemp(prefix=f".{corpus}.", dir=data_root))
    try:
        dataset.save_to_disk(str(temporary))
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"{corpus}: downloaded {HF_DATASETS[corpus]}")


def download_emphassess(data_root: Path, force: bool) -> None:
    destination = data_root / "emphassess"
    if destination.exists() and not force:
        manifest = validate_emphassess_directory(destination)
        print("emphassess: existing validated download; skipping")
        _print_emphassess_counts(manifest)
        return
    if force:
        shutil.rmtree(destination, ignore_errors=True)

    work = Path(tempfile.mkdtemp(prefix=".emphassess.", dir=data_root))
    archive_path = work / "EmphAssess_Dataset.tar.gz.part"
    try:
        urllib.request.urlretrieve(EMPHASSESS_URL, archive_path)
        final_archive = work / "EmphAssess_Dataset.tar.gz"
        archive_path.replace(final_archive)
        with tarfile.open(final_archive, "r:gz") as archive:
            _safe_extract(archive, work)
        # The official archive is flat; tolerate a single wrapper only to locate
        # the files, then normalize the persisted layout to the documented flat form.
        annotation = next(work.rglob("gold_df.json"), None)
        if annotation is None:
            raise FileNotFoundError("gold_df.json was not found after extraction")
        source_root = annotation.parent
        normalized = Path(tempfile.mkdtemp(prefix=".emphassess.normalized.", dir=data_root))
        for path in source_root.iterdir():
            if path.is_file() and path != final_archive:
                shutil.move(str(path), normalized / path.name)
        shutil.move(str(final_archive), normalized / final_archive.name)
        manifest = validate_emphassess_directory(normalized)
        (normalized / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        normalized.replace(destination)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        if "normalized" in locals():
            shutil.rmtree(normalized, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _print_emphassess_counts(manifest)


def _print_emphassess_counts(manifest: dict) -> None:
    print(f"EmphAssess annotations: {manifest['num_annotations']}")
    print(f"EmphAssess WAV files: {manifest['num_audio_files']}")
    print(f"Invalid punctuation-emphasis rows: {manifest['num_filtered_invalid_emphasis']}")
    print(f"Retained EmphAssess rows: {manifest['num_retained_samples']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--corpora", nargs="+", choices=SUPPORTED_CORPORA, default=list(SUPPORTED_CORPORA))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    for corpus in args.corpora:
        if corpus == "emphassess":
            download_emphassess(args.data_root, args.force)
        else:
            download_hf_corpus(corpus, args.data_root, args.force)


if __name__ == "__main__":
    main()
