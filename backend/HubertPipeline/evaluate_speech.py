# evaluate_speech.py
import os
from pydub import AudioSegment
from .syllables import WORD_MAP
from .features_hubert import extract_embedding
from .scorer_hubert import score_syllable
import json

# Load templates
TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "syllable_templates.json"
)

with open(TEMPLATE_PATH) as f:
    templates = json.load(f)

def evaluate(audio_path, word_id):
    syllables = WORD_MAP[word_id]["syllables"]
    audio = AudioSegment.from_wav(audio_path)

    dur = audio.duration_seconds
    syl_dur = dur / len(syllables)

    results = []

    for i, syl in enumerate(syllables):
        start = int(i * syl_dur * 1000)
        end = int((i+1) * syl_dur * 1000)

        temp = f"temp_{syl}.wav"
        audio[start:end].export(temp, format="wav")

        emb = extract_embedding(temp)
        sim, ok = score_syllable(word_id, syl, emb, templates)

        results.append({
            "syllable": syl,
            "similarity": sim,
            "correct": ok
        })

        os.remove(temp)

    return results