"""HuBERT embedding extraction (lazy-loaded, CPU/GPU aware)."""
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import librosa
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

_MODEL_NAME = "facebook/hubert-base-ls960"
_extractor: Wav2Vec2FeatureExtractor | None = None
_model: HubertModel | None = None


def _get_model() -> tuple[Wav2Vec2FeatureExtractor, HubertModel]:
    global _extractor, _model
    if _model is None:
        _extractor = Wav2Vec2FeatureExtractor.from_pretrained(_MODEL_NAME)
        _model = HubertModel.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _extractor, _model


def extract_embedding(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=16000)
    audio, _ = librosa.effects.trim(audio)

    if len(audio) < 2000:
        return np.zeros((768,), dtype=np.float32)

    extractor, model = _get_model()
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state  # (1, T, 768)

    emb = outputs.mean(dim=1).squeeze().numpy()
    emb /= np.linalg.norm(emb) + 1e-8
    return emb.astype(np.float32)
