"""DTW log-mel scoring against pre-computed syllable templates."""
import json
import os

import numpy as np
from dtw import dtw

from app.core.paths import DTW_TEMPLATE_PATH

from .features import extract_features

THRESHOLD = 700.0

_templates: dict | None = None


def _load_templates() -> dict:
    global _templates
    if _templates is None:
        if not DTW_TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"DTW templates not found at {DTW_TEMPLATE_PATH}. "
                "Run `python -m app.scripts.preprocess_dtw` from the backend directory."
            )
        with open(DTW_TEMPLATE_PATH, encoding="utf-8") as f:
            _templates = json.load(f)
    return _templates


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def dtw_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(dtw(a, b, dist=lambda x, y: np.linalg.norm(x - y))[0])


def score_syllable(word_id: str, syl: str, clip_path: str) -> dict:
    templates = _load_templates()
    user = normalize(extract_features(clip_path))

    refs = [
        normalize(np.array(f, dtype=np.float32))
        for f in templates[word_id][syl]
    ]

    if not refs:
        return {"distance": float("inf"), "similarity": 0.0, "correct": False}

    best_dist = min(dtw_dist(user, r) for r in refs)
    similarity = 1.0 - min(best_dist / THRESHOLD, 1.0)

    return {
        "distance": float(best_dist),
        "similarity": float(similarity),
        "correct": best_dist < THRESHOLD,
    }
