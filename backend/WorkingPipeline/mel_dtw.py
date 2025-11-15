# mel_dtw.py

import json
import numpy as np
from dtw import dtw
from .features import extract_features
import os

# Path relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "syllable_templates.json")

with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    templates = json.load(f)

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def dtw_dist(a, b):
    return dtw(a, b, dist=lambda x, y: np.linalg.norm(x - y))[0]

def score_syllable(word_id, syl, clip_path):
    # User feature
    user = normalize(extract_features(clip_path))

    # Reference features (all speaker samples)
    refs = [
        normalize(np.array(f, dtype=np.float32))
        for f in templates[word_id][syl]
    ]

    # Compare user to EACH reference
    dists = [dtw_dist(user, r) for r in refs]

    # Best match
    best_dist = min(dists)

    # --------------------------
    # FIXED THRESHOLD
    # --------------------------
    # Good range based on your logs: 500–600
    threshold = 700        # <-- TUNE HERE

    similarity = 1.0 - min(best_dist / threshold, 1.0)

    correct = best_dist < threshold

    return {
        "distance": float(best_dist),
        "similarity": float(similarity),
        "correct": correct
    }