# features_hubert.py

import torch
import numpy as np
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# Load PyTorch-only HuBERT components
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model.eval()

def extract_embedding(path):
    audio, sr = librosa.load(path, sr=16000)
    audio, _ = librosa.effects.trim(audio)

    if len(audio) < 2000:
        return np.zeros((768,), dtype=np.float32)

    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state  # shape: (1, T, 768)

    emb = outputs.mean(dim=1).squeeze().numpy()
    emb /= (np.linalg.norm(emb) + 1e-8)

    return emb.astype(np.float32)