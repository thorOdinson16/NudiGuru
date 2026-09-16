from datetime import datetime

from pydantic import BaseModel


class SessionOut(BaseModel):
    lesson_id: str
    accuracy: float
    is_battle: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ProgressOut(BaseModel):
    lesson_id: str
    best_accuracy: float
    attempts: int

    model_config = {"from_attributes": True}


class UserStats(BaseModel):
    full_name: str
    email: str
    daily_practice_count: int
    daily_goal: int
    streak_days: int
    total_practices: int
    unique_lessons: int
    avg_accuracy: int
    recent_sessions: list[SessionOut]
