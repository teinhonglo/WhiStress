import numpy as np
from .utils import get_loaded_model, scored_transcription
from typing import Union, Dict


class WhiStressInferenceClient:
    def __init__(self, device="cuda", metadata=None):
        self.device = device
        self.whistress = get_loaded_model(self.device, metadata=metadata)

    def predict(
        self, audio: Dict[str, Union[np.ndarray, int]], transcription=None, return_pairs=True, phone_ids=None, token_pos_ids=None
    ):
        word_emphasis_pairs = scored_transcription(
            audio=audio, 
            model=self.whistress, 
            device=self.device, 
            strip_words=True, 
            transcription=transcription,
            phone_ids=phone_ids,
            token_pos_ids=token_pos_ids
        )
        if return_pairs:
            return word_emphasis_pairs
        # returs transcription str and list of emphasized words
        return " ".join([x[0] for x in word_emphasis_pairs]), [
            x[0] for x in word_emphasis_pairs if x[1] == 1
        ]
