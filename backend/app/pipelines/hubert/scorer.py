"""Cosine-similarity scoring against HuBERT reference embeddings."""
import numpy as np

THRESHOLD = 0.70


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


def score_syllable(word_id: str, syl: str, user_emb: np.ndarray, templates: dict) -> tuple[float, bool]:
    refs = [np.array(r, dtype=np.float32) for r in templates[word_id][syl]]

    if not refs:
        return 0.0, False

    best_sim = max(cosine(user_emb, r) for r in refs)
    return min(best_sim, 1.0), best_sim >= THRESHOLD
