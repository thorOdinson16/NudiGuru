"""Kannada TTS generation and status endpoints."""
import os
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from scipy.io.wavfile import write as scipy_wav_write

from app.core.paths import TTS_CACHE_DIR, ensure_storage_dirs
from app.data.lessons import LESSONS
from app.tts import module as tts

router = APIRouter(prefix="/tts", tags=["tts"])

_WAV_HEADERS = {"Cache-Control": "public, max-age=3600", "Accept-Ranges": "bytes"}


@router.get("/status")
def tts_status() -> dict:
    cached = len(os.listdir(TTS_CACHE_DIR)) if TTS_CACHE_DIR.exists() else 0
    return {
        "available": tts.tts_available(),
        "cache_dir": str(TTS_CACHE_DIR),
        "cached_files": cached,
    }


@router.get("/generate/{word_id}")
async def generate_tts_audio(word_id: str) -> FileResponse:
    if word_id not in LESSONS:
        raise HTTPException(status_code=404, detail="Lesson not found")

    ensure_storage_dirs()
    cache_path = TTS_CACHE_DIR / f"{word_id}.wav"

    if cache_path.exists():
        return FileResponse(str(cache_path), media_type="audio/wav", headers=_WAV_HEADERS)

    if not tts.tts_available():
        raise HTTPException(status_code=503, detail="TTS model not available")

    try:
        audio_array, sample_rate = tts.generate_kannada_audio(
            text=LESSONS[word_id]["text"], speaker_name="female"
        )
        scipy_wav_write(str(cache_path), sample_rate, audio_array)
    except Exception as exc:  # noqa: BLE001
        print(f"TTS error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    return FileResponse(str(cache_path), media_type="audio/wav", headers=_WAV_HEADERS)
