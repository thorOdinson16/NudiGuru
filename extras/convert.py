import os
from pydub import AudioSegment

folder = r"C:\Users\Admin\Documents\NudiGuru\kk hackathon\vaniauntyf"

aac_files = sorted([f for f in os.listdir(folder) if f.endswith(".ogg")], key=lambda x: int(os.path.splitext(x)[0][1]))

for fname in aac_files:
    basename = os.path.splitext(fname)[0]
    if len(basename) > 2:
        number = basename[1] + basename[2]
    else:
        number = basename[1]
    aac_path = os.path.join(folder, fname)
    wav_path = os.path.join(folder, f"{number}.wav")

    audio = AudioSegment.from_file(aac_path, format="ogg")
    audio.export(wav_path, format="wav")

    print(f"Converted {fname} → {number}.wav")