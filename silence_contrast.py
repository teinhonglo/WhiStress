"""Silence-only and silence-contrastive Whisper decoding for WhiStress.

This is an experimental inference entry point.  It deliberately does not alter
the teacher-forced baseline in ``test.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

from corpora import SUPPORTED_CORPORA, load_corpus
from metrics import compute_prf_metrics
from whistress.inference_client.utils import (
    get_loaded_model,
    get_word_emphasis_pairs,
    merge_stressed_tokens,
    prepare_audio,
)


@dataclass
class DecodeResult:
    token_ids: list[int]
    text: str
    contrast_l2_norms: list[float]
    cache_is_independent: bool
    shared_history_verified: bool
    terminated_by_eos: bool


def _json_dump_line(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


class SilenceContrastDecoder:
    """Greedy two-branch decoder with a separate KV cache per audio branch."""

    def __init__(
        self,
        whistress_model,
        alpha: float,
        plausibility_alpha: float,
        max_new_tokens: int,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= plausibility_alpha <= 1.0:
            raise ValueError("plausibility_alpha must be in [0, 1]")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        self.wrapper = whistress_model
        self.model = whistress_model.whisper_model
        self.processor = whistress_model.processor
        self.device = next(self.model.parameters()).device
        self.alpha = alpha
        self.plausibility_alpha = plausibility_alpha
        self.max_new_tokens = max_new_tokens
        self.generation_config = self.model.generation_config

        self.decoder_start_token_id = self.generation_config.decoder_start_token_id
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.eos_token_id = self.generation_config.eos_token_id
        if isinstance(self.eos_token_id, int):
            self.eos_token_ids = {self.eos_token_id}
        else:
            self.eos_token_ids = set(self.eos_token_id or [])

        # Whisper's English checkpoints generally force the no-timestamps token.
        forced = self.generation_config.forced_decoder_ids
        if forced is None:
            forced = self.processor.get_decoder_prompt_ids(
                task="transcribe", no_timestamps=True
            )
        self.forced_decoder_ids = dict(forced or [])
        self.suppress_tokens = set(self.generation_config.suppress_tokens or [])
        self.begin_suppress_tokens = set(
            self.generation_config.begin_suppress_tokens or []
        )
        self.begin_suppress_index = max(self.forced_decoder_ids, default=0) + 1

    def _encode(self, input_features: torch.Tensor):
        return self.model.get_encoder()(input_features=input_features)

    def _forward_step(self, encoder_outputs, token_input, past_key_values):
        output = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=token_input,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return output.logits[:, -1, :].float(), output.past_key_values

    def _apply_generation_masks(
        self, logits: torch.Tensor, sequence_length: int
    ) -> torch.Tensor:
        logits = logits.clone()
        vocab_size = logits.size(-1)
        suppress_tokens = [
            token for token in self.suppress_tokens if 0 <= token < vocab_size
        ]
        if suppress_tokens:
            logits[:, suppress_tokens] = -torch.inf
        begin_suppress_tokens = [
            token for token in self.begin_suppress_tokens if 0 <= token < vocab_size
        ]
        if sequence_length == self.begin_suppress_index and begin_suppress_tokens:
            logits[:, begin_suppress_tokens] = -torch.inf
        forced_token = self.forced_decoder_ids.get(sequence_length)
        if forced_token is not None:
            forced_value = logits[:, forced_token].clone()
            logits.fill_(-torch.inf)
            logits[:, forced_token] = forced_value
        return logits

    def _apply_plausibility(
        self, contrast_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        if self.plausibility_alpha == 0.0:
            return contrast_logits
        threshold = real_logits.amax(dim=-1, keepdim=True) + math.log(
            self.plausibility_alpha
        )
        return contrast_logits.masked_fill(real_logits < threshold, -torch.inf)

    @torch.inference_mode()
    def decode(
        self,
        real_features: torch.Tensor,
        silence_features: torch.Tensor,
        scope: str,
    ) -> DecodeResult:
        if scope not in ("silence", "first_step", "all_steps"):
            raise ValueError(f"Unsupported scope: {scope}")
        if real_features.shape != silence_features.shape:
            raise ValueError(
                "real and silence processor tensors differ: "
                f"{tuple(real_features.shape)} != {tuple(silence_features.shape)}"
            )

        real_encoder = self._encode(real_features)
        silence_encoder = self._encode(silence_features)
        generated = torch.tensor(
            [[self.decoder_start_token_id]], device=self.device, dtype=torch.long
        )
        real_past = None
        silence_past = None
        norms: list[float] = []
        cache_is_independent = True
        shared_history_verified = True
        terminated_by_eos = False
        free_generation_step = 0

        for step in range(self.max_new_tokens):
            token_input = generated if real_past is None else generated[:, -1:]
            # Both calls receive the exact same decoder token history.  Their
            # encoder outputs and KV caches remain branch-local.
            real_token_input = token_input
            silence_token_input = token_input.clone()
            shared_history_verified &= torch.equal(
                real_token_input, silence_token_input
            )
            real_logits, real_past = self._forward_step(
                real_encoder, real_token_input, real_past
            )
            silence_logits, silence_past = self._forward_step(
                silence_encoder, silence_token_input, silence_past
            )
            cache_is_independent &= real_past is not silence_past
            # Record the vocabulary-space difference before common generation
            # masks introduce matching -inf entries.
            norms.append(
                torch.linalg.vector_norm(real_logits - silence_logits).item()
            )

            real_logits = self._apply_generation_masks(real_logits, generated.size(1))
            silence_logits = self._apply_generation_masks(
                silence_logits, generated.size(1)
            )

            is_forced_step = generated.size(1) in self.forced_decoder_ids
            use_contrast = scope == "all_steps" or (
                scope == "first_step" and free_generation_step == 0
            )
            if scope == "silence":
                selection_logits = silence_logits
            elif use_contrast:
                selection_logits = (
                    (1.0 + self.alpha) * real_logits
                    - self.alpha * silence_logits
                )
                selection_logits = self._apply_plausibility(
                    selection_logits, real_logits
                )
            else:
                selection_logits = real_logits

            next_token = selection_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
            if not is_forced_step:
                free_generation_step += 1
            if next_token.item() in self.eos_token_ids:
                terminated_by_eos = True
                break

        token_ids = generated[0].tolist()
        text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        return DecodeResult(
            token_ids=token_ids,
            text=text.strip(),
            contrast_l2_norms=norms,
            cache_is_independent=cache_is_independent,
            shared_history_verified=shared_history_verified,
            terminated_by_eos=terminated_by_eos,
        )

    @torch.inference_mode()
    def stress_predictions(
        self, input_features: torch.Tensor, token_ids: Iterable[int]
    ) -> list[dict[str, Any]]:
        decoder_ids = torch.tensor(
            [list(token_ids)], device=self.device, dtype=torch.long
        )
        output = self.wrapper(
            input_features=input_features,
            decoder_input_ids=decoder_ids,
        )
        stress_ids = output.preds
        # WhiStress labels are left-shifted with respect to decoder tokens.
        stress_ids = torch.cat((stress_ids[:, -1:], stress_ids[:, :-1]), dim=1)
        pairs = get_word_emphasis_pairs(
            decoder_ids[0], stress_ids[0], self.processor, filter_special_tokens=True
        )
        return [
            {"word": word.strip(), "stress": int(stress)}
            for word, stress in merge_stressed_tokens(pairs)
            if word.strip()
        ]


def _prediction_record(sample, result: DecodeResult, stress) -> dict[str, Any]:
    return {
        "text_id": str(sample["id"]),
        "audio": sample["audio"].get("path"),
        "reference": sample["transcription"],
        "prediction": result.text,
        "token_ids": result.token_ids,
        "stress": stress,
        "terminated_by_eos": result.terminated_by_eos,
    }


def _evaluate_prediction_file(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    references: list[int] = []
    predictions: list[int] = []
    exact_transcripts = 0
    stress_evaluable = 0
    for row in rows:
        exact_transcripts += int(
            row["prediction"].strip().lower() == row["reference"].strip().lower()
        )
        pred_stress = [item["stress"] for item in row["stress"]]
        # Reference labels are included by the decoding stage.
        ref_stress = row.get("reference_stress", [])
        if len(pred_stress) == len(ref_stress):
            stress_evaluable += 1
            predictions.extend(pred_stress)
            references.extend(ref_stress)
    return {
        "num_samples": len(rows),
        "generation_failure_rate": (
            sum(
                (not row["prediction"]) or (not row["terminated_by_eos"])
                for row in rows
            ) / len(rows)
            if rows
            else 0.0
        ),
        "max_token_termination_rate": (
            sum(not row["terminated_by_eos"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "exact_transcript_rate": exact_transcripts / len(rows) if rows else 0.0,
        "stress_coverage_rate": stress_evaluable / len(rows) if rows else 0.0,
        "stress_metrics": (
            compute_prf_metrics(predictions, references) if references else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "evaluate"), default="decode")
    parser.add_argument("--pretrained_ckpt_dir", type=Path, required=True)
    parser.add_argument("--corpus", choices=SUPPORTED_CORPORA, default="tinystress")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--plausibility_alpha", type=float, default=0.0)
    parser.add_argument(
        "--contrast_scope",
        choices=("first_step", "all_steps", "both"),
        default="all_steps",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "evaluate":
        metrics = {}
        for scope in ("silence", "first_step", "all_steps"):
            path = args.results_dir / f"predictions_{scope}.jsonl"
            if path.is_file():
                metrics[scope] = _evaluate_prediction_file(path)
        (args.results_dir / "silence_contrast_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        return

    metadata_path = args.pretrained_ckpt_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Preserve the project's loader while making the checkpoint selected on the
    # command line authoritative if metadata was created on another machine.
    metadata["path_to_weights"] = str(args.pretrained_ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_loaded_model(device=device, metadata=metadata)
    if args.dtype != "auto":
        dtype = getattr(torch, args.dtype)
        if device.type == "cpu" and dtype == torch.float16:
            raise ValueError("float16 inference on CPU is not supported")
        torch.nn.Module.to(model, device=device, dtype=dtype)
    decoder = SilenceContrastDecoder(
        model,
        alpha=args.alpha,
        plausibility_alpha=args.plausibility_alpha,
        max_new_tokens=args.max_new_tokens,
    )
    dataset = load_corpus(args.corpus, args.split, args.data_root / "raw")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    scopes = ["silence"]
    scopes += (
        ["first_step", "all_steps"]
        if args.contrast_scope == "both"
        else [args.contrast_scope]
    )
    handles = {
        scope: (args.results_dir / f"predictions_{scope}.jsonl").open("w")
        for scope in scopes
    }
    analysis_handle = (args.results_dir / "contrast_analysis.jsonl").open("w")
    try:
        for sample in tqdm(dataset, desc="Silence contrast decoding"):
            source_audio = sample["audio"]
            source_array = np.asarray(source_audio["array"], dtype=np.float32)
            source_silence = np.zeros_like(source_array)
            assert source_silence.shape == source_array.shape

            if np.max(np.abs(source_array), initial=0.0) == 0.0:
                # Existing prepare_audio normalizes by the peak and would divide
                # an already-silent source by zero.
                target_samples = round(
                    source_array.size * 16000 / source_audio["sampling_rate"]
                )
                real_audio = np.zeros(target_samples, dtype=np.float32)
            else:
                real_audio = prepare_audio(source_audio)
            silence_audio = np.zeros_like(real_audio)
            real_features = model.processor.feature_extractor(
                real_audio, sampling_rate=16000, return_tensors="pt"
            )["input_features"].to(device)
            silence_features = model.processor.feature_extractor(
                silence_audio, sampling_rate=16000, return_tensors="pt"
            )["input_features"].to(device)
            if real_features.shape != silence_features.shape:
                raise RuntimeError("Processor output shape mismatch")

            results = {}
            stress_outputs = {}
            for scope in scopes:
                result = decoder.decode(real_features, silence_features, scope)
                branch_features = silence_features if scope == "silence" else real_features
                stress = decoder.stress_predictions(
                    branch_features, result.token_ids
                )
                record = _prediction_record(sample, result, stress)
                record["reference_stress"] = [
                    int(index in sample["emphasis_indices"])
                    for index, _ in enumerate(sample["transcription"].split())
                ]
                _json_dump_line(handles[scope], record)
                results[scope] = result
                stress_outputs[scope] = stress

            analysis = {
                "text_id": str(sample["id"]),
                "audio": source_audio.get("path"),
                "num_samples": int(source_array.size),
                "sampling_rate": int(source_audio["sampling_rate"]),
                "duration_sec": source_array.size / source_audio["sampling_rate"],
                "alpha": args.alpha,
                "plausibility_alpha": args.plausibility_alpha,
                "processor_shapes": {
                    "real": list(real_features.shape),
                    "silence": list(silence_features.shape),
                },
                "silence_audio_verification": {
                    "num_samples": int(source_silence.size),
                    "sampling_rate": int(source_audio["sampling_rate"]),
                    "max_abs_waveform": float(
                        np.max(np.abs(source_silence), initial=0.0)
                    ),
                },
                "outputs": {
                    scope: results[scope].text if scope in results else None
                    for scope in ("silence", "first_step", "all_steps")
                },
                "stress_outputs": stress_outputs,
                "logit_contrast_l2_norm_by_step": {
                    scope: (
                        results[scope].contrast_l2_norms if scope in results else []
                    )
                    for scope in ("first_step", "all_steps")
                },
                "verification": {
                    scope: {
                        "independent_kv_cache": results[scope].cache_is_independent,
                        "shared_generated_history": results[scope].shared_history_verified,
                    }
                    for scope in scopes
                },
            }
            _json_dump_line(analysis_handle, analysis)
    finally:
        for handle in handles.values():
            handle.close()
        analysis_handle.close()


if __name__ == "__main__":
    main()
