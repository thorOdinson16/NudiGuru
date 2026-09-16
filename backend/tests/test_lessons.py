from app.data.lessons import LESSONS, difficulty_for, lesson_order, serialize_lesson


def test_all_lessons_have_required_fields():
    for word_id, data in LESSONS.items():
        assert data["text"], word_id
        assert data["roman"], word_id
        assert data["english"], word_id
        assert data["syllables"], word_id


def test_serialize_lesson_fields():
    lesson = serialize_lesson("w01")
    assert lesson["id"] == "w01"
    assert lesson["order"] == 1
    assert lesson["difficulty"] == "beginner"
    assert lesson["kannada_text"] == lesson["title"]
    assert lesson["transliteration"] == "Namaste"


def test_difficulty_boundaries():
    assert difficulty_for("w05") == "beginner"
    assert difficulty_for("w10") == "intermediate"
    assert difficulty_for("w15") == "advanced"


def test_lesson_order():
    assert lesson_order("w07") == 7
