# preprocess_references.py

import os
import json
from pydub import AudioSegment
from tqdm import tqdm
from syllables import WORD_MAP
from features import extract_features

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REFERENCE_DIR = "Voices/"

templates = {}

# ----------------------------
# Build templates
# ----------------------------
for word_id, info in tqdm(WORD_MAP.items(), desc="Words"):
    syllables = info["syllables"]
    templates[word_id] = {s: [] for s in syllables}

    speakers = [
        s for s in os.listdir(REFERENCE_DIR)
        if os.path.isdir(os.path.join(REFERENCE_DIR, s))
    ]

    for speaker in speakers:
        wav_path = os.path.join(REFERENCE_DIR, speaker, f"{int(word_id[1:])}.wav")
        if not os.path.exists(wav_path):
            continue

        audio = AudioSegment.from_wav(wav_path)

        duration = audio.duration_seconds
        syl_dur = duration / len(syllables)

        # Slice evenly by syllable count
        for i, syl in enumerate(syllables):
            start = int(i * syl_dur * 1000)
            end = int((i + 1) * syl_dur * 1000)

            temp = f"temp_{speaker}_{word_id}_{syl}.wav"
            audio[start:end].export(temp, format="wav")

            feat = extract_features(temp)
            templates[word_id][syl].append(feat.tolist())

            os.remove(temp)

# ----------------------------
# Save templates
# ----------------------------
with open("syllable_templates.json", "w") as f:
    json.dump(templates, f, indent=2)

print("DONE → syllable_templates.json")