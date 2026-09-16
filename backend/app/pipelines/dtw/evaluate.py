"""DTW pipeline entry point: score each syllable of an uploaded clip."""
import os
import tempfile
import warnings

from pydub import AudioSegment

from app.data.lessons import get_lesson

from .mel_dtw import score_syllable

warnings.filterwarnings("ignore")


def evaluate(audio_path: str, word_id: str) -> list[dict]:
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

            res = score_syllable(word_id, syl, clip_path)
            results.append(
                {
                    "syllable": syl,
                    "distance": res["distance"],
                    "similarity": res["similarity"],
                    "correct": res["correct"],
                }
            )

    return results
