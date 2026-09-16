"""Regenerate HuBERT syllable templates from reference voices.

Run from the backend directory:  python -m app.scripts.preprocess_hubert
"""
import json
import os
import tempfile

from pydub import AudioSegment
from tqdm import tqdm

from app.core.paths import HUBERT_TEMPLATE_PATH, VOICES_DIR
from app.data.lessons import LESSONS
from app.pipelines.hubert.features import extract_embedding


def main() -> None:
    if not VOICES_DIR.exists():
        raise SystemExit(f"Reference voices directory not found: {VOICES_DIR}")

    speakers = sorted(d for d in os.listdir(VOICES_DIR) if (VOICES_DIR / d).is_dir())
    templates: dict = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for word_id, info in tqdm(LESSONS.items(), desc="Words"):
            syllables = info["syllables"]
            templates[word_id] = {s: [] for s in syllables}

            for speaker in speakers:
                wav_path = VOICES_DIR / speaker / f"{int(word_id[1:])}.wav"
                if not wav_path.exists():
                    continue

                audio = AudioSegment.from_wav(str(wav_path))
                syl_dur = audio.duration_seconds / max(len(syllables), 1)

                for i, syl in enumerate(syllables):
                    start = int(i * syl_dur * 1000)
                    end = int((i + 1) * syl_dur * 1000)

                    clip = os.path.join(tmpdir, f"{speaker}_{word_id}_{i}.wav")
                    audio[start:end].export(clip, format="wav")
                    templates[word_id][syl].append(extract_embedding(clip).tolist())

    HUBERT_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HUBERT_TEMPLATE_PATH, "w", encoding="utf-8") as f:
        json.dump(templates, f)

    print(f"Wrote {HUBERT_TEMPLATE_PATH}")


if __name__ == "__main__":
    main()
