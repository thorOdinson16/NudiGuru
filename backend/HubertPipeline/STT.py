# STT.py

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

DURATION = 3
SAMPLING_RATE = 16000

def record_audio(filename="speech.wav"):
    print("ದಯವಿಟ್ಟು ಮಾತಾಡಿ... Recording...")
    audio = sd.rec(int(DURATION * SAMPLING_RATE),
                   samplerate=SAMPLING_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()
    write(filename, SAMPLING_RATE, (audio * 32767).astype(np.int16))
    print("Audio saved:", filename)
    return filename

if __name__ == "__main__":
    audio_file = record_audio()