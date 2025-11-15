# scorer_hubert.py
import numpy as np

def cosine(a, b):
    """Cosine similarity between two vectors"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b + 1e-8))

def score_syllable(word_id, syl, user_emb, templates):
    """
    Score a syllable by comparing user embedding to reference templates
    """
    refs = [np.array(r, dtype=np.float32) for r in templates[word_id][syl]]
    
    if len(refs) == 0:
        return 0.0, False
    
    sims = [cosine(user_emb, r) for r in refs]
    best_sim = max(sims)
    
    threshold = 0.70  # Adjust based on testing
    
    # Convert to percentage-like similarity
    similarity = min(best_sim, 1.0)
    
    return similarity, best_sim >= threshold