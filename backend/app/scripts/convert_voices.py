"""Convert reference voice files (e.g. .ogg) into word-id named .wav files.

The pipelines expect reference recordings named by word id (``w01.wav``,
``w02.wav`` ...) inside each speaker folder under ``Voices/``.

Usage:
    python -m app.scripts.convert_voices --folder "path/to/voices" --ext .ogg
"""
import argparse
import os
import re

from pydub import AudioSegment


def convert(folder: str, ext: str) -> None:
    files = sorted(
        (f for f in os.listdir(folder) if f.lower().endswith(ext.lower())),
        key=lambda name: int(re.sub(r"\D", "", os.path.splitext(name)[0]) or 0),
    )

    for fname in files:
        number = re.sub(r"\D", "", os.path.splitext(fname)[0])
        if not number:
            continue
        audio = AudioSegment.from_file(os.path.join(folder, fname))
        out_name = f"w{int(number):02d}.wav"
        audio.export(os.path.join(folder, out_name), format="wav")
        print(f"Converted {fname} -> {out_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert voice files to numbered WAVs")
    parser.add_argument("--folder", required=True, help="Folder containing the source files")
    parser.add_argument("--ext", default=".ogg", help="Source extension (default: .ogg)")
    args = parser.parse_args()
    convert(args.folder, args.ext)


if __name__ == "__main__":
    main()
