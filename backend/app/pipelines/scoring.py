"""Pipeline availability and combined scoring."""
from app.core.paths import DTW_TEMPLATE_PATH, HUBERT_TEMPLATE_PATH
from app.data.lessons import get_lesson


def dtw_available() -> bool:
    return DTW_TEMPLATE_PATH.exists()


def hubert_available() -> bool:
    return HUBERT_TEMPLATE_PATH.exists()


def _run_dtw(audio_path: str, word_id: str) -> dict:
    from .dtw import evaluate

    results = evaluate(audio_path, word_id)
    accuracy = int(sum(r["similarity"] for r in results) / max(len(results), 1) * 100)
    return {
        "accuracy": accuracy,
        "syllables": [
            {
                "text": r["syllable"],
                "accuracy": int(r["similarity"] * 100),
                "distance": r.get("distance", 0.0),
                "correct": r.get("correct", False),
            }
            for r in results
        ],
    }


def _run_hubert(audio_path: str, word_id: str) -> dict:
    from .hubert import evaluate

    results = evaluate(audio_path, word_id)
    accuracy = int(sum(r["similarity"] for r in results) / max(len(results), 1) * 100)
    return {
        "accuracy": accuracy,
        "syllables": [
            {
                "text": r["syllable"],
                "accuracy": int(r["similarity"] * 100),
                "correct": r.get("correct", False),
            }
            for r in results
        ],
    }


def evaluate_all(audio_path: str, word_id: str) -> dict:
    """Run every available pipeline and combine the results."""
    detailed: dict = {}
    errors: dict = {}

    if dtw_available():
        try:
            detailed["dtw_pipeline"] = _run_dtw(audio_path, word_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            errors["dtw_pipeline"] = str(exc)

    if hubert_available():
        try:
            detailed["hubert_pipeline"] = _run_hubert(audio_path, word_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            errors["hubert_pipeline"] = str(exc)

    if not detailed:
        raise RuntimeError(
            "No scoring pipeline is available. "
            "Generate templates with the preprocess scripts."
        )

    combined = _combine(word_id, detailed)

    return {
        "accuracy_score": combined["accuracy_score"],
        "syllables": combined["syllables"],
        "detailed_results": detailed,
        "errors": errors,
    }


def _combine(word_id: str, detailed: dict) -> dict:
    pipelines = [v for v in detailed.values() if "accuracy" in v]
    accuracy = int(sum(p["accuracy"] for p in pipelines) / len(pipelines))

    syllable_texts = get_lesson(word_id)["syllables"]
    combined_syllables = []
    for i, text in enumerate(syllable_texts):
        scores = [
            p["syllables"][i]["accuracy"]
            for p in pipelines
            if i < len(p["syllables"])
        ]
        combined_syllables.append(
            {
                "text": text,
                "accuracy": int(sum(scores) / len(scores)) if scores else 0,
            }
        )

    return {"accuracy_score": accuracy, "syllables": combined_syllables}
