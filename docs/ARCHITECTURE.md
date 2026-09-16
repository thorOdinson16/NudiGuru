# NudiGuru Architecture

## Overview

NudiGuru is a Kannada pronunciation-learning app. The React frontend records a
learner's voice, uploads it to the FastAPI backend, and receives syllable-level
pronunciation feedback. Reference audio is produced by a Kannada TTS model.

```
Browser (React/Vite)
   │  recorded WAV + lesson_id
   ▼
FastAPI (backend/app)
   ├── /auth/*        JWT auth (bcrypt + PyJWT)
   ├── /lessons       lesson catalogue (static data)
   ├── /evaluate      dual-pipeline pronunciation scoring
   ├── /tts/*         FastPitch + HiFi-GAN reference audio
   └── /user/*        stats & progress from PostgreSQL
          │
          ├── DTW pipeline   (log-mel + Dynamic Time Warping)
          ├── HuBERT pipeline (transformer embeddings + cosine similarity)
          └── PostgreSQL (SQLAlchemy async + Alembic)
```

## Backend layout (`backend/app`)

| Path | Responsibility |
| --- | --- |
| `main.py` | FastAPI app, CORS, lifespan, router wiring, `/` and `/health` |
| `core/config.py` | Settings from environment / `.env` (pydantic-settings) |
| `core/paths.py` | All filesystem paths resolved relative to the package |
| `core/security.py` | Password hashing and JWT create/decode |
| `db/` | SQLAlchemy async engine, session, ORM models |
| `api/` | Routers: `auth`, `lessons`, `evaluate`, `tts`, `user`, plus `deps` |
| `schemas/` | Pydantic request/response models |
| `data/lessons.py` | **Single source of truth** for lessons |
| `data/templates/` | Generated DTW/HuBERT syllable templates (gitignored) |
| `pipelines/dtw/` | Log-mel feature extraction, DTW scoring, evaluation |
| `pipelines/hubert/` | HuBERT embeddings, cosine scoring, evaluation |
| `pipelines/scoring.py` | Runs all available pipelines and combines results |
| `tts/module.py` | Lazy wrapper around the IndicTTS engine |
| `tts/engine/` | FastPitch/HiFi-GAN inference engine (text normalization, VAD) |
| `scripts/` | Offline tools: preprocess templates, convert voices |

## Pronunciation scoring

Both pipelines split the uploaded clip into equal-length syllable slices
(one per syllable in the lesson) and score each slice.

- **DTW pipeline** — pre-emphasis, silence trim, 40-bin log-mel spectrogram,
  CMVN normalization, then DTW distance against reference templates. A syllable
  is correct when the best distance is below `700`; similarity is
  `1 - min(distance / 700, 1)`.
- **HuBERT pipeline** — `facebook/hubert-base-ls960` embeddings (mean-pooled,
  L2-normalized), cosine similarity against reference embeddings. A syllable is
  correct when the best cosine similarity is at least `0.70`.

`pipelines/scoring.py` averages the available pipelines into a combined
accuracy and combined per-syllable scores. If a pipeline's templates are
missing it is simply skipped; if none are available the request fails with
`503`.

## Data & storage

- **PostgreSQL** stores users, practice sessions, and per-lesson progress.
  Schema is managed with Alembic (`backend/alembic`).
- **`backend/kn/`** holds the TTS weights (not committed).
- **`backend/Voices/`** holds reference speaker recordings (committed,
  `<Speaker>/w01.wav` … `w15.wav`).
- **`backend/storage/`** holds transient uploads and cached TTS audio
  (not committed). Uploads are deleted after each request.
- Generated templates (`data/templates/*.json`) are derived from `Voices/` and
  are not committed.

## Configuration

All backend settings come from environment variables or `backend/.env`
(see `backend/.env.example`): `DATABASE_URL`, `JWT_SECRET`, `JWT_ALGORITHM`,
`ACCESS_TOKEN_EXPIRE_MINUTES`, `CORS_ORIGINS`, `TTS_DEVICE`, `ENABLE_DENOISER`.

The frontend reads `VITE_API_URL` (see `frontend/.env.example`).

## Battle mode

Battle mode is **client-side**: two players record on the same device and each
attempt is scored through the normal `/evaluate` endpoint. There is no
server-side matchmaking or room state.
