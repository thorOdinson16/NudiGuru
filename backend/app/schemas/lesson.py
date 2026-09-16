from pydantic import BaseModel


class LessonOut(BaseModel):
    id: str
    order: int
    title: str
    kannada_text: str
    transliteration: str
    english_translation: str
    syllables: list[str]
    difficulty: str
