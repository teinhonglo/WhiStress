import json
import wave

import pytest

from corpora import (
    InvalidEmphasisError,
    adapt_emphassess_row,
    adapt_expresso_example,
    adapt_stress_benchmark_example,
    adapt_tinystress_example,
    compute_stress_binary,
    is_expresso_evaluation_example,
    parse_expresso_emphasis,
    validate_canonical_sample,
    validate_emphassess_directory,
)


def write_wav(path):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16)


def audio():
    return {"array": [0.0] * 4, "sampling_rate": 16000, "path": None}


def benchmark():
    return {
        "transcription_id": "sentence",
        "interpretation_id": "interpretation",
        "transcription": "we really agree",
        "audio": audio(),
        "stress_pattern": {"binary": [0, 1, 0], "indices": [1], "words": ["really"]},
    }


def test_tinystress_adapter():
    sample = adapt_tinystress_example({"id": 7, "transcription": "hello world", "audio": audio(), "emphasis_indices": [1]})
    assert sample["id"] == "7" and sample["source_dataset"] == "tinystress"


def test_stresstest_adapter():
    sample = adapt_stress_benchmark_example(benchmark(), "stresstest")
    assert sample["id"] == "interpretation" and sample["emphasis_indices"] == [1]


def test_stresspresso_adapter():
    example = benchmark()
    example["metadata"] = {"speaker_id": "speaker"}
    assert adapt_stress_benchmark_example(example, "stresspresso")["source_dataset"] == "stresspresso"


def test_stress_benchmark_binary_consistency():
    example = benchmark()
    example["stress_pattern"]["binary"] = [1, 0, 0]
    with pytest.raises(ValueError, match="interpretation"):
        adapt_stress_benchmark_example(example, "stresstest")


def test_expresso_adapter_removes_markers_and_maps_multiword_span():
    sample = adapt_expresso_example({
        "id": "ex01_default_00001",
        "text": "Was *this element* always *there?*",
        "speaker_id": "ex01",
        "style": "default",
        "audio": audio(),
    })
    assert sample["transcription"] == "Was this element always there?"
    assert sample["emphasis_indices"] == [1, 2, 4]
    assert sample["source_dataset"] == "expresso"


def test_expresso_parser_handles_marker_only_tokens_and_external_punctuation():
    transcription, indices = parse_expresso_emphasis("Maybe * you should * change *priorities*, okay.")
    assert transcription == "Maybe you should change priorities, okay."
    assert indices == [1, 2, 4]


def test_expresso_evaluation_filter_matches_paper_protocol():
    assert is_expresso_evaluation_example("I said *now*.", "ex01")
    assert is_expresso_evaluation_example("I said *now*.", "ex02")
    assert not is_expresso_evaluation_example("I said *now*.", "ex03")
    assert not is_expresso_evaluation_example("I said it now.", "ex01")


def test_expresso_rejects_unbalanced_markers():
    with pytest.raises(ValueError, match="Unbalanced"):
        parse_expresso_emphasis("I said *right now")


def test_expresso_adapter_rejects_sample_outside_evaluation_subset():
    with pytest.raises(ValueError, match="ex01/ex02"):
        adapt_expresso_example({
            "id": "ex03_default_00001",
            "text": "I said *now*.",
            "speaker_id": "ex03",
            "style": "default",
            "audio": audio(),
        })


def test_emphassess_punctuation_remapping(tmp_path):
    write_wav(tmp_path / "item.wav")
    sample = adapt_emphassess_row({"id": "item", "src_sentence": ["hello", ",", "world", "!"], "gold_emphasis": [2], "voice": "v"}, tmp_path)
    assert sample["transcription"] == "hello world" and sample["emphasis_indices"] == [1]
    assert compute_stress_binary(sample["transcription"], sample["emphasis_indices"]) == [0, 1]


def test_emphassess_keeps_internal_apostrophe(tmp_path):
    write_wav(tmp_path / "item.wav")
    sample = adapt_emphassess_row({"id": "item", "src_sentence": ["today's", "plan", "."], "gold_emphasis": [0]}, tmp_path)
    assert sample["transcription"] == "today's plan"


def test_emphassess_filters_punctuation_emphasis(tmp_path):
    with pytest.raises(InvalidEmphasisError):
        adapt_emphassess_row({"id": "item", "src_sentence": ["hello", "!"], "gold_emphasis": [1]}, tmp_path)


def test_emphassess_audio_path_uses_flat_layout(tmp_path):
    write_wav(tmp_path / "item.wav")
    sample = adapt_emphassess_row({"id": "item", "src_sentence": ["hello"], "gold_emphasis": []}, tmp_path)
    assert sample["audio"]["path"] == str(tmp_path / "item.wav")


def test_missing_audio_detection(tmp_path):
    with pytest.raises(FileNotFoundError):
        adapt_emphassess_row({"id": "missing", "src_sentence": ["hello"], "gold_emphasis": []}, tmp_path)


def test_duplicate_id_detection(tmp_path):
    row = {"id": "same", "src_sentence": ["hello"], "gold_emphasis": []}
    (tmp_path / "gold_df.json").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    write_wav(tmp_path / "same.wav")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_emphassess_directory(tmp_path, require_official_counts=False)


def test_invalid_audio_format_detection(tmp_path):
    row = {"id": "item", "src_sentence": ["hello"], "gold_emphasis": []}
    (tmp_path / "gold_df.json").write_text(json.dumps(row) + "\n")
    with wave.open(str(tmp_path / "item.wav"), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 4)
    with pytest.raises(ValueError, match="mono PCM16"):
        validate_emphassess_directory(tmp_path, require_official_counts=False)


def test_canonical_index_range_validation():
    with pytest.raises(ValueError, match="out-of-range"):
        validate_canonical_sample({"id": "bad", "transcription": "one word", "emphasis_indices": [2]})
