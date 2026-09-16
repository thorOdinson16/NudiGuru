"""Lesson listing endpoints."""
from fastapi import APIRouter

from app.data.lessons import LESSONS, serialize_lesson
from app.schemas.lesson import LessonOut

router = APIRouter(tags=["lessons"])


@router.get("/lessons", response_model=list[LessonOut])
def get_lessons() -> list[dict]:
    return sorted((serialize_lesson(word_id) for word_id in LESSONS), key=lambda x: x["order"])
