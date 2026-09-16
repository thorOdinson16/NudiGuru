"""HuBERT pipeline entry point: score each syllable of an uploaded clip."""
import json
import os
import tempfile

from pydub import AudioSegment

from app.core.paths import HUBERT_TEMPLATE_PATH
from app.data.lessons import get_lesson

from .features import extract_embedding
from .scorer import score_syllable

_templates: dict | None = None


def _load_templates() -> dict:
    global _templates
    if _templates is None:
        if not HUBERT_TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"HuBERT templates not found at {HUBERT_TEMPLATE_PATH}. "
                "Run `python -m app.scripts.preprocess_hubert` from the backend directory."
            )
        with open(HUBERT_TEMPLATE_PATH, encoding="utf-8") as f:
            _templates = json.load(f)
    return _templates


def evaluate(audio_path: str, word_id: str) -> list[dict]:
    templates = _load_templates()
    syllables = get_lesson(word_id)["syllables"]
    audio = AudioSegment.from_wav(audio_path)

    duration = audio.duration_seconds
    syl_dur = duration / max(len(syllables), 1)

    results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, syl in enumerate(syllables):
            start = int(i * syl_dur * 1000)
            end = int((i + 1) * syl_dur * 1000)

            clip_path = os.path.join(tmpdir, f"{i}_{syl}.wav")
            audio[start:end].export(clip_path, format="wav")

            emb = extract_embedding(clip_path)
            sim, ok = score_syllable(word_id, syl, emb, templates)

            results.append(
                {
                    "syllable": syl,
                    "similarity": float(sim),
                    "correct": bool(ok),
                }
            )

    return results
