# evaluate_speech.py

import os
from pydub import AudioSegment
from .syllables import WORD_MAP
from .mel_dtw import score_syllable
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evaluate(audio_path, word_id):
    syllables = WORD_MAP[word_id]["syllables"]
    audio = AudioSegment.from_wav(audio_path)

    duration = audio.duration_seconds
    syl_dur = duration / len(syllables)

    results = []

    for i, syl in enumerate(syllables):
        start = int(i * syl_dur * 1000)
        end = int((i + 1) * syl_dur * 1000)

        temp = f"temp_{syl}.wav"
        audio[start:end].export(temp, format="wav")

        res = score_syllable(word_id, syl, temp)

        results.append({
            "syllable": syl,
            "distance": res["distance"],
            "similarity": res["similarity"],
            "correct": res["correct"]
        })

        os.remove(temp)

    return results