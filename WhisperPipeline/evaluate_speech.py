# evaluate_speech.py

import os
import json
from pydub import AudioSegment
from features_whisper import extract_embedding
from scorer_whisper import score_syllable
from syllables import WORD_MAP
import warnings
warnings.filterwarnings("ignore")

with open("syllable_templates.json") as f:
    templates = json.load(f)

def evaluate(audio_path, word_id):
    syllables = WORD_MAP[word_id]["syllables"]
    audio = AudioSegment.from_wav(audio_path)

    dur = audio.duration_seconds
    syl_dur = dur / len(syllables)

    results = []

    for i, syl in enumerate(syllables):
        start = int(i * syl_dur * 1000)
        end = int((i + 1) * syl_dur * 1000)

        temp = f"temp_{syl}.wav"
        audio[start:end].export(temp, format="wav")

        user_emb = extract_embedding(temp)
        sim, correct = score_syllable(word_id, syl, user_emb, templates)

        results.append({
            "syllable": syl,
            "similarity": sim,
            "correct": correct
        })

        os.remove(temp)

    return results