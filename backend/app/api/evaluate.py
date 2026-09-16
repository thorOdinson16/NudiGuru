"""Pronunciation evaluation endpoint."""
import os
import tempfile
import traceback

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.paths import UPLOAD_DIR, ensure_storage_dirs
from app.data.lessons import LESSONS, get_lesson
from app.db.models import LessonProgress, PracticeSession, User
from app.db.session import get_db
from app.pipelines import scoring
from app.schemas.evaluate import EvaluationResponse

from .deps import get_current_user_optional

router = APIRouter(tags=["evaluate"])

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
WEAK_SYLLABLE_THRESHOLD = 70


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pronunciation(
    audio: UploadFile = File(...),
    lesson_id: str = Form(...),
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
) -> EvaluationResponse:
    if lesson_id not in LESSONS:
        raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")

    if not (audio.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files accepted")

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large")

    ensure_storage_dirs()
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".wav", dir=str(UPLOAD_DIR)
    )
    tmp.write(content)
    tmp.close()
    temp_path = tmp.name

    try:
        result = scoring.evaluate_all(temp_path, lesson_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"Evaluation error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    accuracy = result["accuracy_score"]
    syllables = result["syllables"]

    await _persist(user, db, lesson_id, accuracy)

    weak = [s for s in syllables if s["accuracy"] < WEAK_SYLLABLE_THRESHOLD]
    tips = (
        [f"Focus on '{s['text']}'" for s in weak[:3]]
        if weak
        else ["Excellent pronunciation!"]
    )

    return EvaluationResponse(
        accuracy_score=accuracy,
        syllables=syllables,
        areas_to_improve=tips,
        reference_audio_url=f"/tts/generate/{lesson_id}",
        detailed_results=result["detailed_results"],
    )


async def _persist(
    user: User | None, db: AsyncSession, lesson_id: str, accuracy: int
) -> None:
    if user is None:
        return

    try:
        db.add(
            PracticeSession(user_id=user.id, lesson_id=lesson_id, accuracy=float(accuracy))
        )

        result = await db.execute(
            select(LessonProgress).where(
                LessonProgress.user_id == user.id,
                LessonProgress.lesson_id == lesson_id,
            )
        )
        progress = result.scalar_one_or_none()
        if progress is None:
            db.add(
                LessonProgress(
                    user_id=user.id,
                    lesson_id=lesson_id,
                    best_accuracy=float(accuracy),
                    attempts=1,
                )
            )
        else:
            progress.attempts += 1
            progress.best_accuracy = max(progress.best_accuracy, float(accuracy))

        await db.commit()
    except Exception:  # noqa: BLE001 - never fail a valid evaluation on a DB hiccup
        await db.rollback()
        print(f"Failed to persist practice session: {traceback.format_exc()}")
