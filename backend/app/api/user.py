"""User statistics derived from persisted practice sessions."""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LessonProgress, PracticeSession, User
from app.db.session import get_db
from app.schemas.user import ProgressOut, SessionOut, UserStats

from .deps import get_current_user

router = APIRouter(prefix="/user", tags=["user"])

DAILY_GOAL = 10


@router.get("/stats", response_model=UserStats)
async def get_user_stats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserStats:
    result = await db.execute(
        select(PracticeSession)
        .where(PracticeSession.user_id == user.id)
        .order_by(PracticeSession.created_at.desc())
    )
    sessions = result.scalars().all()

    total = len(sessions)
    unique_lessons = len({s.lesson_id for s in sessions})
    avg_accuracy = int(sum(s.accuracy for s in sessions) / total) if total else 0

    today = datetime.now(timezone.utc).date()
    daily_count = sum(1 for s in sessions if s.created_at.date() == today)

    streak_days = _compute_streak({s.created_at.date() for s in sessions}, today)

    return UserStats(
        full_name=user.full_name or user.email.split("@")[0],
        email=user.email,
        daily_practice_count=daily_count,
        daily_goal=DAILY_GOAL,
        streak_days=streak_days,
        total_practices=total,
        unique_lessons=unique_lessons,
        avg_accuracy=avg_accuracy,
        recent_sessions=[SessionOut.model_validate(s) for s in sessions[:5]],
    )


def _compute_streak(days: set, today) -> int:
    """Consecutive days with practice, counting back from today (or yesterday)."""
    if not days:
        return 0

    cursor = today if today in days else today - timedelta(days=1)
    streak = 0
    while cursor in days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


@router.get("/progress", response_model=list[ProgressOut])
async def get_progress(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ProgressOut]:
    result = await db.execute(
        select(LessonProgress).where(LessonProgress.user_id == user.id)
    )
    return [ProgressOut.model_validate(p) for p in result.scalars().all()]
