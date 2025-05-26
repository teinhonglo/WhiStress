import torch
from transformers import WhisperConfig
import librosa
import numpy as np
import pathlib
from torch.nn import functional as F
from ..model import WhiStress


PATH_TO_WEIGHTS = pathlib.Path(__file__).parent.parent / "weights"


def get_loaded_model(device="cuda"):
    whisper_model_name = f"openai/whisper-small.en"
    whisper_config = WhisperConfig()
    whistress_model = WhiStress(
        whisper_config, layer_for_head=9, whisper_backbone_name=whisper_model_name
    ).to(device)
    whistress_model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]
    whistress_model.load_model(PATH_TO_WEIGHTS)
    whistress_model.to(device)
    whistress_model.eval()
    return whistress_model


def get_word_emphasis_pairs(
    transcription_preds, emphasis_preds, processor, filter_special_tokens=True
):
    emphasis_preds_list = emphasis_preds.tolist()
    transcription_preds_words = [
        processor.tokenizer.decode([i], skip_special_tokens=False)
        for i in transcription_preds
    ]
    if filter_special_tokens:
        special_tokens_indices = [
            i
            for i, x in enumerate(transcription_preds)
            if x in processor.tokenizer.all_special_ids
        ]
        emphasis_preds_list = [
            x
            for i, x in enumerate(emphasis_preds_list)
            if i not in special_tokens_indices
        ]
        transcription_preds_words = [
            x
            for i, x in enumerate(transcription_preds_words)
            if i not in special_tokens_indices
        ]
    return list(zip(transcription_preds_words, emphasis_preds_list))


def inference_from_audio(audio: np.ndarray, model: WhiStress, device: str):
    input_features = model.processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    )["input_features"]
    out_model = model.generate_dual(input_features=input_features.to(device))
    emphasis_probs = F.softmax(out_model.logits, dim=-1)
    emphasis_preds = torch.argmax(emphasis_probs, dim=-1)
    emphasis_preds_right_shifted = torch.cat((emphasis_preds[:, -1:], emphasis_preds[:, :-1]), dim=1)
    word_emphasis_pairs = get_word_emphasis_pairs(
        out_model.preds[0],
        emphasis_preds_right_shifted[0],
        model.processor,
        filter_special_tokens=True,
    )
    return word_emphasis_pairs


def prepare_audio(audio, target_sr=16000):
    # resample to 16kHz
    sr = audio["sampling_rate"]
    y = audio["array"]
    y = np.array(y, dtype=float)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # Normalize the audio (scale to [-1, 1])
    y_resampled /= max(abs(y_resampled))
    return y_resampled


def merge_stressed_tokens(tokens_with_stress):
    """
    tokens_with_stress is a list of tuples: (token_string, stress_value)
    e.g.:
       [(" I", 0), (" didn", 1), ("'t", 0), (" say", 0), (" he", 0), (" stole", 0),
        (" the", 0), (" money", 0), (".", 0)]
    Returns a list of merged tuples, combining subwords into full words.
    """
    merged = []

    current_word = ""
    current_stress = 0  # 0 means not stressed, 1 means stressed

    for token, stress in tokens_with_stress:
        # If token starts with a space (or is the very first), we treat it as a new word
        # or if current_word is empty (first iteration).
        if token.startswith(" ") or current_word == "":
            # If we already have something in current_word, push it into merged
            # before starting a new one
            if current_word:
                merged.append((current_word, current_stress))

            # Start a new word
            current_word = token
            current_stress = stress
        else:
            # Otherwise, it's a subword that should be appended to the previous word
            current_word += token
            # If any sub-token is stressed, the whole merged word is stressed
            current_stress = max(current_stress, stress)

    # Don't forget to append the final word
    if current_word:
        merged.append((current_word, current_stress))

    return merged


def inference_from_audio_and_transcription(
    audio: np.ndarray, transcription, model: WhiStress, device: str
):
    input_features = model.processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    )["input_features"]
    # convert transcription to input_ids
    input_ids = model.processor.tokenizer(
        transcription,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=30,
    )["input_ids"]
    out_model = model(
                    input_features=input_features.to(device),
                    decoder_input_ids=input_ids.to(device),
                )
    emphasis_probs = F.softmax(out_model.logits, dim=-1)
    emphasis_preds = torch.argmax(emphasis_probs, dim=-1)
    emphasis_preds_right_shifted = torch.cat((emphasis_preds[:, -1:], emphasis_preds[:, :-1]), dim=1)
    word_emphasis_pairs = get_word_emphasis_pairs(
        input_ids[0],
        emphasis_preds_right_shifted[0],
        model.processor,
        filter_special_tokens=True,
    )
    return word_emphasis_pairs

def scored_transcription(audio, model, strip_words=True, transcription: str = None, device="cuda"):
    audio_arr = prepare_audio(audio)
    token_stress_pairs = None
    if transcription: # if we want to use the ground truth transcription
        token_stress_pairs = inference_from_audio_and_transcription(audio_arr, transcription, model, device)
    else:
        token_stress_pairs = inference_from_audio(audio_arr, model, device)
    # token_stress_pairs = inference_from_audio(audio_arr, model)
    word_level_stress = merge_stressed_tokens(token_stress_pairs)
    if strip_words:
        word_level_stress = [(word.strip(), stress) for word, stress in word_level_stress]
    return word_level_stress
