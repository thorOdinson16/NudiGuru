# scorer_whisper.py

import numpy as np

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def score_syllable(word_id, syl, emb_user, templates):
    refs = [np.array(r, dtype=np.float32) for r in templates[word_id][syl]]

    # Similarity to all 8 reference speakers
    sims = [cosine(emb_user, r) for r in refs]
    best = max(sims)

    # Threshold logic:
    # 8 speakers → natural variation → good pronunciation ~0.70–0.85
    threshold = 0.70

    return best, best >= threshold