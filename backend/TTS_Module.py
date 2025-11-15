# backend/TTS.py
import io
import numpy as np
from TTS.utils.synthesizer import Synthesizer
from src.inference import TextToSpeechEngine
from scipy.io.wavfile import write as scipy_wav_write

# ---------------------------
# Load Kannada IndicTTS Model
# ---------------------------

kannada_model = Synthesizer(
    tts_checkpoint="kn/fastpitch/best_model.pth",
    tts_config_path="kn/fastpitch/config.json",
    tts_speakers_file="kn/fastpitch/speakers.pth",
    tts_languages_file=None,
    vocoder_checkpoint="kn/hifigan/best_model.pth",
    vocoder_config="kn/hifigan/config.json",
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=False
)

# Set up engine
models = {
    "kn": kannada_model
}

engine = TextToSpeechEngine(models)

DEFAULT_SAMPLING_RATE = 16000

# ---------------------------
# Helper Function
# ---------------------------

def generate_kannada_audio(text, speaker_name="female"):
    """Generate Kannada TTS audio"""
    print(f"🎤 Generating TTS: '{text}' with {speaker_name} voice")
    
    kannada_raw_audio = engine.infer_from_text(
        input_text=text,
        lang="kn",
        speaker_name=speaker_name
    )
    
    # Validate output
    if kannada_raw_audio is None or len(kannada_raw_audio) == 0:
        raise ValueError("TTS engine returned empty audio")
    
    # Convert to numpy array
    audio_array = np.array(kannada_raw_audio, dtype=np.float32)
    
    print(f"✅ Generated {len(audio_array)} samples at {DEFAULT_SAMPLING_RATE}Hz")
    
    return audio_array, DEFAULT_SAMPLING_RATE