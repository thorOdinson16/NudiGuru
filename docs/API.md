# NudiGuru API

Base URL: `http://localhost:8000` (configurable via `VITE_API_URL` on the
frontend).

Authenticated endpoints expect a bearer token:
`Authorization: Bearer <access_token>`.

## Status

### `GET /`
Returns service status and component availability.

```json
{
  "status": "running",
  "lessons": 15,
  "dtw_pipeline": true,
  "hubert_pipeline": true,
  "tts_model": true
}
```

### `GET /health`
Returns `{ "status": "ok", ...component availability }`.

## Auth

### `POST /auth/register`
Body: `{ "email": string, "password": string (min 6), "full_name": string }`
Returns `201`: `{ "access_token": string, "token_type": "bearer", "user": {...} }`
Errors: `409` if the email is already registered.

### `POST /auth/login`
Body: `{ "email": string, "password": string }`
Returns `200` with the same shape as register. Errors: `401` invalid credentials.

### `GET /auth/me` — auth required
Returns the current user: `{ "id", "email", "full_name" }`.

## Lessons

### `GET /lessons`
Returns the lesson catalogue sorted by order:

```json
[
  {
    "id": "w01",
    "order": 1,
    "title": "ನಮಸ್ತೆ",
    "kannada_text": "ನಮಸ್ತೆ",
    "transliteration": "Namaste",
    "english_translation": "Hello / Greetings",
    "syllables": ["na", "mas", "te"],
    "difficulty": "beginner"
  }
]
```

## Evaluation

### `POST /evaluate`
`multipart/form-data`:
- `audio` — WAV file (max 15 MB)
- `lesson_id` — e.g. `w01`

Auth is optional. When a bearer token is supplied the result is persisted to the
user's history.

Returns:

```json
{
  "accuracy_score": 82,
  "syllables": [{ "text": "na", "accuracy": 90 }],
  "areas_to_improve": ["Focus on 'mas'"],
  "reference_audio_url": "/tts/generate/w01",
  "detailed_results": { "dtw_pipeline": {}, "hubert_pipeline": {} },
  "stt_rejected": false,
  "reason": null
}
```

Errors: `400` non-WAV/empty, `404` unknown lesson, `413` too large,
`503` no scoring pipeline available.

## Text-to-speech

### `GET /tts/generate/{word_id}`
Returns a WAV file (generated on first request, then cached).
Errors: `404` unknown lesson, `503` model not available.

### `GET /tts/status`
```json
{ "available": true, "cache_dir": "...", "cached_files": 3 }
```

## User — auth required

### `GET /user/stats`
```json
{
  "full_name": "Learner",
  "email": "learner@example.com",
  "daily_practice_count": 3,
  "daily_goal": 10,
  "streak_days": 4,
  "total_practices": 42,
  "unique_lessons": 7,
  "avg_accuracy": 81,
  "recent_sessions": [
    { "lesson_id": "w02", "accuracy": 88, "is_battle": false, "created_at": "..." }
  ]
}
```

### `GET /user/progress`
Returns per-lesson best scores:
`[{ "lesson_id": "w01", "best_accuracy": 91.0, "attempts": 3 }]`
