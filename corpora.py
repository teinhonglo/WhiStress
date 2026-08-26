"""Corpus adapters for WhiStress evaluation datasets."""

from __future__ import annotations

import json
import string
import wave
from array import array
from pathlib import Path
from typing import Any

SUPPORTED_CORPORA = (
    "tinystress",
    "stresstest",
    "stresspresso",
    "expresso",
    "emphassess",
)
EXPRESSO_EVAL_SPEAKERS = frozenset(("ex01", "ex02"))


class InvalidEmphasisError(ValueError):
    """An emphasis label points at a standalone punctuation token."""


def is_standalone_punctuation(token: str) -> bool:
    return bool(token) and all(character in string.punctuation for character in token)


def validate_canonical_sample(sample: dict[str, Any]) -> dict[str, Any]:
    words = sample["transcription"].strip().split()
    if not words:
        raise ValueError(f"Sample {sample.get('id')!r} has an empty transcription")
    invalid = [index for index in sample["emphasis_indices"] if not 0 <= index < len(words)]
    if invalid:
        raise ValueError(
            f"Sample {sample.get('id')!r} has out-of-range emphasis indices {invalid} "
            f"for {len(words)} words"
        )
    return sample


def compute_stress_binary(transcription: str, emphasis_indices: list[int]) -> list[int]:
    """Build the word-level labels consumed by ``StressDataset.preprocess``."""
    words = transcription.strip().split()
    return [1 if index in emphasis_indices else 0 for index in range(len(words))]


def _canonical_audio(audio: dict[str, Any]) -> dict[str, Any]:
    return {
        "array": audio["array"],
        "sampling_rate": int(audio["sampling_rate"]),
        "path": audio.get("path"),
    }


def adapt_tinystress_example(example: dict[str, Any]) -> dict[str, Any]:
    return validate_canonical_sample({
        "id": str(example["id"]),
        "transcription": example["transcription"],
        "audio": _canonical_audio(example["audio"]),
        "emphasis_indices": list(example["emphasis_indices"]),
        "source_dataset": "tinystress",
    })


def adapt_stress_benchmark_example(
    example: dict[str, Any], source_dataset: str
) -> dict[str, Any]:
    if source_dataset not in ("stresstest", "stresspresso"):
        raise ValueError(f"Unsupported stress benchmark: {source_dataset}")
    pattern = example["stress_pattern"]
    sample = validate_canonical_sample({
        "id": str(example["interpretation_id"]),
        "transcription": example["transcription"],
        "audio": _canonical_audio(example["audio"]),
        "emphasis_indices": list(pattern["indices"]),
        "source_dataset": source_dataset,
    })
    expected_binary = compute_stress_binary(
        sample["transcription"], sample["emphasis_indices"]
    )
    if list(pattern["binary"]) != expected_binary:
        raise ValueError(f"Stress-pattern binary mismatch for sample {sample['id']}")
    return sample


def parse_expresso_emphasis(text: str) -> tuple[str, list[int]]:
    """Remove Expresso's ``*...*`` markup and return stressed word indices."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expresso text must be a non-empty string")

    words: list[str] = []
    emphasis_indices: list[int] = []
    inside_emphasis = False
    for marked_word in text.strip().split():
        clean_characters: list[str] = []
        word_is_emphasized = False
        for character in marked_word:
            if character == "*":
                inside_emphasis = not inside_emphasis
                continue
            clean_characters.append(character)
            if inside_emphasis:
                word_is_emphasized = True

        clean_word = "".join(clean_characters)
        # Marker-only tokens are allowed, for example ``* several words *``.
        if not clean_word:
            continue
        words.append(clean_word)
        if word_is_emphasized:
            emphasis_indices.append(len(words) - 1)

    if inside_emphasis:
        raise ValueError(f"Unbalanced Expresso emphasis markers: {text!r}")
    if not words:
        raise ValueError("Expresso text contains no words after removing emphasis markers")
    return " ".join(words), emphasis_indices


def is_expresso_evaluation_example(text: str, speaker_id: str) -> bool:
    """Implement the ex01/ex02, positive-stress protocol used by prior work."""
    if speaker_id not in EXPRESSO_EVAL_SPEAKERS:
        return False
    _, emphasis_indices = parse_expresso_emphasis(text)
    return bool(emphasis_indices)


def adapt_expresso_example(example: dict[str, Any]) -> dict[str, Any]:
    transcription, emphasis_indices = parse_expresso_emphasis(example["text"])
    if example["speaker_id"] not in EXPRESSO_EVAL_SPEAKERS or not emphasis_indices:
        raise ValueError(
            "Expresso evaluation keeps only ex01/ex02 samples with at least one "
            "asterisk-marked stressed word"
        )
    return validate_canonical_sample({
        "id": str(example["id"]),
        "transcription": transcription,
        "audio": _canonical_audio(example["audio"]),
        "emphasis_indices": emphasis_indices,
        "source_dataset": "expresso",
        "speaker_id": example["speaker_id"],
        "style": example.get("style"),
    })


def _read_wav(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing EmphAssess audio: {path}")
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sampling_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if channels != 1 or sample_width != 2 or sampling_rate != 16000:
        raise ValueError(f"Expected 16 kHz mono PCM16 WAV: {path}")
    samples = array("h")
    samples.frombytes(frames)
    if samples.itemsize != sample_width:
        raise ValueError(f"Unsupported PCM sample width in WAV: {path}")
    return {
        "array": [sample / 32768.0 for sample in samples],
        "sampling_rate": sampling_rate,
        "path": str(path),
    }


def adapt_emphassess_row(row: dict[str, Any], emphassess_root: str | Path) -> dict[str, Any]:
    tokens = list(row["src_sentence"])
    emphasis = list(row["gold_emphasis"])
    if any(index < 0 or index >= len(tokens) for index in emphasis):
        raise ValueError(f"EmphAssess row {row.get('id')!r} has an invalid original index")
    if any(is_standalone_punctuation(tokens[index]) for index in emphasis):
        raise InvalidEmphasisError(
            f"EmphAssess row {row['id']} emphasizes standalone punctuation"
        )

    old_to_new: dict[int, int] = {}
    words = []
    for old_index, token in enumerate(tokens):
        if not is_standalone_punctuation(token):
            old_to_new[old_index] = len(words)
            words.append(token)
    audio_path = Path(emphassess_root) / f"{row['id']}.wav"
    return validate_canonical_sample({
        "id": str(row["id"]),
        "transcription": " ".join(words),
        "audio": _read_wav(audio_path),
        "emphasis_indices": [old_to_new[index] for index in emphasis],
        "source_dataset": "emphassess",
        "voice": row.get("voice"),
    })


def _read_emphassess_rows(root: Path) -> tuple[list[dict[str, Any]], int]:
    annotation_path = root / "gold_df.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing EmphAssess annotations: {annotation_path}")
    rows = [json.loads(line) for line in annotation_path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate EmphAssess annotation ID detected")
    samples = []
    filtered = 0
    for row in rows:
        try:
            samples.append(adapt_emphassess_row(row, root))
        except InvalidEmphasisError:
            filtered += 1
    return samples, filtered


def load_corpus(corpus: str, split: str = "test", data_root: str | Path = "data/raw"):
    from datasets import Dataset, DatasetDict, load_from_disk

    if corpus not in SUPPORTED_CORPORA:
        raise ValueError(f"Unsupported corpus {corpus!r}; choose from {SUPPORTED_CORPORA}")
    root = Path(data_root) / corpus
    if corpus == "emphassess":
        # A complete official download must never be evaluated with silently
        # changed annotation, audio, or filtering counts.
        validate_emphassess_directory(root)
        samples, filtered = _read_emphassess_rows(root)
        dataset = Dataset.from_list(samples)
        dataset.info.description = json.dumps({
            "num_original_samples": len(samples) + filtered,
            "num_filtered_invalid_emphasis": filtered,
            "num_retained_samples": len(samples),
        })
        return dataset

    stored = load_from_disk(str(root))
    if isinstance(stored, DatasetDict):
        if split in stored:
            raw = stored[split]
        elif corpus == "expresso" and list(stored.keys()) == ["train"]:
            # The public Expresso ``read`` configuration exposes one source
            # split. It is filtered below into the established SSD test set.
            raw = stored["train"]
        else:
            raise KeyError(
                f"Split {split!r} is unavailable for {corpus!r}; "
                f"available splits: {list(stored.keys())}"
            )
    else:
        raw = stored
    if corpus == "tinystress":
        return raw.map(adapt_tinystress_example, remove_columns=raw.column_names)
    if corpus == "expresso":
        num_original_samples = len(raw)
        selected = raw.filter(
            is_expresso_evaluation_example,
            input_columns=["text", "speaker_id"],
            desc="Selecting the Expresso SSD evaluation subset",
        )
        dataset = selected.map(
            adapt_expresso_example,
            remove_columns=selected.column_names,
        )
        dataset.info.description = json.dumps({
            "num_original_samples": num_original_samples,
            "num_filtered_by_protocol": num_original_samples - len(dataset),
            "num_retained_samples": len(dataset),
            "source_split": "train",
            "evaluation_speakers": sorted(EXPRESSO_EVAL_SPEAKERS),
            "requires_positive_stress": True,
        })
        return dataset
    return raw.map(
        lambda example: adapt_stress_benchmark_example(example, corpus),
        remove_columns=raw.column_names,
    )


def validate_emphassess_directory(root: str | Path, require_official_counts: bool = True) -> dict[str, int | str]:
    root = Path(root)
    rows = [json.loads(line) for line in (root / "gold_df.json").read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate EmphAssess annotation ID detected")
    wav_stems = {path.stem for path in root.glob("*.wav")}
    if set(ids) != wav_stems:
        missing = sorted(set(ids) - wav_stems)
        orphaned = sorted(wav_stems - set(ids))
        raise ValueError(f"EmphAssess audio mismatch: missing={missing[:5]}, orphaned={orphaned[:5]}")
    for wav_path in root.glob("*.wav"):
        with wave.open(str(wav_path), "rb") as wav_file:
            audio_format = (
                wav_file.getframerate(),
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getcomptype(),
            )
        if audio_format != (16000, 1, 2, "NONE"):
            raise ValueError(
                f"Expected 16 kHz mono PCM16 WAV, got {audio_format}: {wav_path}"
            )
    invalid = sum(
        any(0 <= index < len(row["src_sentence"]) and is_standalone_punctuation(row["src_sentence"][index])
            for index in row["gold_emphasis"])
        for row in rows
    )
    manifest = {
        "dataset": "emphassess",
        "num_annotations": len(rows),
        "num_audio_files": len(wav_stems),
        "num_filtered_invalid_emphasis": invalid,
        "num_retained_samples": len(rows) - invalid,
    }
    if require_official_counts and (len(rows), len(wav_stems), invalid, len(rows) - invalid) != (3652, 3652, 12, 3640):
        raise ValueError(f"Unexpected EmphAssess counts: {manifest}")
    return manifest
