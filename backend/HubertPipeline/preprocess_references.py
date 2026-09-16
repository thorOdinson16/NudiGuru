# preprocess_references.py

import os, json
from pydub import AudioSegment
from tqdm import tqdm
from syllables import WORD_MAP
from features_hubert import extract_embedding
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

REFERENCE_DIR = "Voices/"
templates = {}

for word_id, info in tqdm(WORD_MAP.items(), desc="Words"):
    syllables = info["syllables"]
    templates[word_id] = {s: [] for s in syllables}

    speakers = [s for s in os.listdir(REFERENCE_DIR)
                if os.path.isdir(os.path.join(REFERENCE_DIR, s))]

    for spk in speakers:
        wav = os.path.join(REFERENCE_DIR, spk, f"{int(word_id[1:])}.wav")
        if not os.path.exists(wav):
            continue

        audio = AudioSegment.from_wav(wav)
        dur = audio.duration_seconds
        syl_dur = dur / len(syllables)

        for i, syl in enumerate(syllables):
            start = int(i * syl_dur * 1000)
            end = int((i+1) * syl_dur * 1000)

            temp = f"temp_{spk}_{word_id}_{syl}.wav"
            audio[start:end].export(temp, format="wav")

            emb = extract_embedding(temp)
            templates[word_id][syl].append(emb.tolist())

            os.remove(temp)

with open("syllable_templates.json", "w") as f:
    json.dump(templates, f, indent=2)

print("DONE → syllable_templates.json")