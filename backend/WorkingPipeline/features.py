# features.py

import librosa
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_features(path):
    # Load audio
    y, sr = librosa.load(path, sr=16000)

    # ----------------------------------------
    # 1. Pre-emphasis (boost high frequencies)
    # ----------------------------------------
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # ----------------------------------------
    # 2. Trim silence (important!)
    # ----------------------------------------
    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < 0.1 * sr:  # too short fallback
        return np.zeros((10, 40), dtype=np.float32)

    # ----------------------------------------
    # 3. Log-Mel Spectrogram (40 bins)
    # ----------------------------------------
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=400,        # 25 ms
        hop_length=160,   # 10 ms
        win_length=400,
        n_mels=40,
        fmin=20,
        fmax=7600
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    # ----------------------------------------
    # 4. CMVN Normalization
    # ----------------------------------------
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)

    # ----------------------------------------
    # 5. Return (time, melbins)
    # ----------------------------------------
    return logmel.T.astype(np.float32)