"""FastAPI application factory for NudiGuru."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, evaluate, lessons, tts, user
from app.core.config import get_settings
from app.core.paths import (
    DTW_TEMPLATE_PATH,
    HUBERT_TEMPLATE_PATH,
    MODELS_DIR,
    ensure_storage_dirs,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage_dirs()

    try:
        from app.tts import module as tts_module

        if tts_module.tts_available():
            tts_module.get_engine()
            print("TTS engine loaded")
        else:
            print("TTS weights not found; /tts endpoints will report unavailable")
    except Exception as exc:  # noqa: BLE001 - TTS is optional at startup
        print(f"TTS init skipped: {exc}")

    yield


app = FastAPI(title="NudiGuru API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length", "Content-Type"],
)

app.include_router(auth.router)
app.include_router(lessons.router)
app.include_router(evaluate.router)
app.include_router(tts.router)
app.include_router(user.router)


def _component_status() -> dict:
    return {
        "dtw_pipeline": DTW_TEMPLATE_PATH.exists(),
        "hubert_pipeline": HUBERT_TEMPLATE_PATH.exists(),
        "tts_model": (MODELS_DIR / "fastpitch" / "best_model.pth").exists(),
    }


@app.get("/")
def root() -> dict:
    from app.data.lessons import LESSONS

    return {"status": "running", "lessons": len(LESSONS), **_component_status()}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **_component_status()}
