from pydantic import BaseModel


class SyllableScore(BaseModel):
    text: str
    accuracy: int
    distance: float | None = None
    correct: bool | None = None


class EvaluationResponse(BaseModel):
    accuracy_score: int
    syllables: list[SyllableScore]
    areas_to_improve: list[str]
    reference_audio_url: str
    detailed_results: dict
    stt_rejected: bool = False
    reason: str | None = None
