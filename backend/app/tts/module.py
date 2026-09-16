"""Kannada TTS wrapper around the IndicTTS FastPitch + HiFi-GAN engine.

The model is loaded lazily on first use so the API can start without the
(~1.6 GB) weights present (TTS endpoints will report unavailable instead).
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_settings
from app.core.paths import MODELS_DIR

DEFAULT_SAMPLING_RATE = 16000

_engine = None


def _build_engine():
    # Imported lazily so the API can boot without the heavy TTS stack installed.
    from TTS.utils.synthesizer import Synthesizer

    from .engine.inference import TextToSpeechEngine

    settings = get_settings()
    use_cuda = settings.tts_device.lower() == "cuda"

    fastpitch = MODELS_DIR / "fastpitch"
    hifigan = MODELS_DIR / "hifigan"

    synthesizer = Synthesizer(
        tts_checkpoint=str(fastpitch / "best_model.pth"),
        tts_config_path=str(fastpitch / "config.json"),
        tts_speakers_file=str(fastpitch / "speakers.pth"),
        tts_languages_file=None,
        vocoder_checkpoint=str(hifigan / "best_model.pth"),
        vocoder_config=str(hifigan / "config.json"),
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=use_cuda,
    )

    return TextToSpeechEngine(
        {"kn": synthesizer},
        enable_denoiser=settings.enable_denoiser,
    )


def get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def tts_available() -> bool:
    return (MODELS_DIR / "fastpitch" / "best_model.pth").exists() and (
        MODELS_DIR / "hifigan" / "best_model.pth"
    ).exists()


def generate_kannada_audio(text: str, speaker_name: str = "female") -> tuple[np.ndarray, int]:
    """Generate Kannada speech, always returned at DEFAULT_SAMPLING_RATE."""
    engine = get_engine()
    raw = engine.infer_from_text(input_text=text, lang="kn", speaker_name=speaker_name)

    if raw is None or len(raw) == 0:
        raise ValueError("TTS engine returned empty audio")

    audio = np.asarray(raw, dtype=np.float32)
    if engine.target_sr != DEFAULT_SAMPLING_RATE:
        import librosa

        audio = librosa.resample(
            audio, orig_sr=engine.target_sr, target_sr=DEFAULT_SAMPLING_RATE
        )

    return audio, DEFAULT_SAMPLING_RATE
