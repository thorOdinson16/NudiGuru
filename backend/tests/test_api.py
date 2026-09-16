from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_status():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["lessons"] == 15


def test_health():
    assert client.get("/health").status_code == 200


def test_list_lessons():
    response = client.get("/lessons")
    assert response.status_code == 200
    lessons = response.json()
    assert len(lessons) == 15
    assert lessons[0]["id"] == "w01"
    assert lessons[0]["order"] == 1


def test_evaluate_unknown_lesson():
    response = client.post(
        "/evaluate",
        files={"audio": ("test.wav", b"RIFF....WAVE", "audio/wav")},
        data={"lesson_id": "nope"},
    )
    assert response.status_code == 404


def test_evaluate_rejects_non_wav():
    response = client.post(
        "/evaluate",
        files={"audio": ("test.mp3", b"data", "audio/mpeg")},
        data={"lesson_id": "w01"},
    )
    assert response.status_code == 400
