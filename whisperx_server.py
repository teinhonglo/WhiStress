from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import string, soundfile as sf
import uvicorn
import json

from local.e2e_stt.whisperx_models import SpeechModel
from local.e2e_stt.audio_models import AudioModel

import numpy as np
from starlette.responses import Response

speech_model = SpeechModel(tag="small.en", device="cuda", language="en",
                           condition_on_previous_text=False, args=None)
audio_model  = AudioModel(sample_rate=16000)

app = FastAPI(title="whisperx-delivery")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def np_dumps(obj) -> str:
    return json.dumps(obj, cls=NpEncoder, ensure_ascii=False)

class Req(BaseModel):
    wav_path: str
    transcription: Optional[str] = None

@app.post("/extract")
def extract(req: Req):
    try:
        speech, sr = sf.read(req.wav_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"read wav failed: {e}")

    # === 你的邏輯 ===
    _, f0_info     = audio_model.get_f0(speech)
    _, energy_info = audio_model.get_energy(speech)

    tx = req.transcription.translate(str.maketrans('', '', string.punctuation.replace("'", "").replace("-", ""))) if req.transcription else None
    text_result, ctm_results = speech_model.recog(req.wav_path, text_prompt=tx)
    word_ctm_info, phn_ctm_info = ctm_results

    payload = {
        "word_ctm": word_ctm_info,
        "feats": {**f0_info, **energy_info}
    }
    return Response(content=np_dumps(payload), media_type="application/json")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6208, workers=1)

