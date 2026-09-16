import librosa
import numpy as np
from dtw import dtw

# ---------------------------------------
# Function: Compare two audio files
# ---------------------------------------
def compare_audio(file1, file2, threshold=300):
    # Load both audio files (same directory)
    y1, sr1 = librosa.load(file1, sr=16000)
    y2, sr2 = librosa.load(file2, sr=16000)

    # Extract MFCC features
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

    # Run DTW
    dist, cost, acc, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: np.linalg.norm(x - y))

    print(f"DTW Distance: {dist}")

    # Decide similar or different
    if dist < threshold:
        print("✅ The two spoken words are SIMILAR")
    else:
        print("❌ The two spoken words are DIFFERENT")

    return dist

# ---------------------------------------
# Example usage
# ---------------------------------------
file1 = "speech1.wav"
file2 = "speech2.wav"
file3 = "speech3.wav"
file4 = "speech4.wav"
file5 = "speech5.wav"
file6 = "speech6.wav"
file7 = "speech7.wav"
file8 = "speech8.wav"
file9 = "speech9.wav"
file10 = "speech10.wav"
file11 = "speech11.wav"
file12 = "speech12.wav"

compare_audio(file1, file2) # Namaskara, Bengaluru
compare_audio(file3, file4) # Dhanyavaada, Dharmadeesha
compare_audio(file5, file6) # Dayavittu Sahaya, Dayaavittu Sahaya
compare_audio(file7, file8) # Bhanu, Surya
compare_audio(file9, file10) # Namaskara, na ma sa ka ra
compare_audio(file11, file12) # Namaskara, Namaskara