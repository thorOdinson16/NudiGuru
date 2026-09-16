# features_whisper.py

import whisper
import numpy as np
import torch

MODEL = whisper.load_model("base")   # or "small"

def extract_embedding(path):
    # 1. Load and pad/trim like Whisper expects
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)

    # 2. Whisper’s OWN mel extractor (CRITICAL)
    mel = whisper.log_mel_spectrogram(audio)  # shape: (80, 3000)

    # 3. Encoder forward pass
    with torch.no_grad():
        encoded = MODEL.encoder(mel.unsqueeze(0))   # shape: (1, T, 512)

    # 4. Average across time → final 512-dim embedding
    emb = encoded.mean(dim=1).squeeze().cpu().numpy()

    return emb.astype(np.float32)