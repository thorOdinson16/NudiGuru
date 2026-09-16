# NudiGuru Setup

## Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 13+
- `ffmpeg` on PATH (used by `pydub` for audio slicing)

## Backend

From the `backend/` directory:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

`requirements.lock.txt` captures the exact versions of a fully verified working
environment (including the ML stack). Use it if you need a reproducible install:

```bash
pip install -r requirements.lock.txt
```

If you have a CUDA GPU, install a matching PyTorch build for faster HuBERT/TTS
inference and set `TTS_DEVICE=cuda` in `.env`.

### 3. Configure environment

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

Edit `.env` and set `DATABASE_URL`, `JWT_SECRET`, and `CORS_ORIGINS`.
Default `DATABASE_URL`:
`postgresql+asyncpg://postgres:postgres@localhost:5432/nudiguru`.

### 4. Create the database and run migrations

```bash
createdb nudiguru           # or use psql/pgAdmin
alembic upgrade head
```

### 5. Provide the TTS model (~1.6 GB)

The Kannada FastPitch + HiFi-GAN weights are not committed. Download them from
the project's GitHub Releases and extract so the layout is:

```
backend/kn/fastpitch/best_model.pth
backend/kn/fastpitch/config.json
backend/kn/fastpitch/speakers.pth
backend/kn/hifigan/best_model.pth
backend/kn/hifigan/config.json
```

Without these, the API still starts and reports `tts_model: false`; the
`/tts/*` endpoints return `503`.

### 6. Provide reference voices and templates

See [Reference voices](#reference-voices) below. The scoring pipelines need the
generated templates in `backend/app/data/templates/`.

### 7. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check: <http://localhost:8000/health>

## Frontend

From the `frontend/` directory:

```bash
npm install
copy .env.example .env      # set VITE_API_URL if the backend is not on :8000
npm run dev
```

The app runs at <http://localhost:5173>. Register an account on the Sign Up
page — authentication is required to access the app routes.

## Reference voices

Reference recordings live under `backend/Voices/<Speaker>/<wordId>.wav`, e.g.
`backend/Voices/PriyaF/1.wav` corresponds to lesson `w01`. Each speaker folder
should contain one WAV per lesson id (`1.wav` … `15.wav`).

To convert source recordings (e.g. `.ogg`) into the expected numbered WAVs:

```bash
python -m app.scripts.convert_voices --folder "path/to/speaker" --ext .ogg
```

Then regenerate the syllable templates (run from `backend/`):

```bash
python -m app.scripts.preprocess_dtw
python -m app.scripts.preprocess_hubert
```

These write `backend/app/data/templates/dtw.json` and `hubert.json`
(not committed). HuBERT preprocessing downloads `facebook/hubert-base-ls960`
(~360 MB) on first run and needs 4 GB+ RAM.

## Tests

```bash
cd backend
pytest
```

`tests/test_api.py` runs the FastAPI app via the test client; the DB-dependent
endpoints are not exercised there, so a running PostgreSQL is not required for
the current suite.

## Troubleshooting

- **`503` from `/evaluate`** — templates missing; run the preprocess scripts.
- **`503` from `/tts/*`** — TTS weights missing or `TTS_DEVICE` misconfigured.
- **`401` on `/user/*`** — missing/expired token; sign in again.
- **`502`/connection errors from the frontend** — check `VITE_API_URL` and that
  the backend is running; CORS only allows origins in `CORS_ORIGINS`.
- **Audio slicing errors** — install `ffmpeg` and ensure it is on PATH.
