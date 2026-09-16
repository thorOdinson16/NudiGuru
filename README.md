# NudiGuru – AI-Powered Kannada Pronunciation Learning Platform

NudiGuru helps learners speak Kannada accurately with real-time, syllable-level
pronunciation feedback, TTS reference audio, and progress tracking.

The app records a learner's voice, scores it with two independent speech
pipelines, and highlights the syllables that need work.

---

## Features

- **Dual-pipeline pronunciation scoring**
  - **DTW pipeline** — log-mel spectrogram + Dynamic Time Warping
  - **HuBERT pipeline** — `facebook/hubert-base-ls960` embeddings + cosine similarity
  - Combined accuracy with per-syllable scores and targeted tips
- **Kannada TTS** — FastPitch + HiFi-GAN reference audio (cached per lesson)
- **Lessons** — 15 curated Kannada phrases with transliteration, English meaning,
  and syllable breakdown
- **Practice** — record, evaluate, and review syllable-level feedback
- **Battle mode** — two players compete on the same device (client-side)
- **Accounts & progress** — JWT auth, PostgreSQL-backed practice history and stats

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Backend | FastAPI, SQLAlchemy (async), Alembic, PostgreSQL |
| Auth | bcrypt + PyJWT |
| Speech ML | PyTorch, Transformers (HuBERT), librosa, dtw |
| TTS | IndicTTS FastPitch + HiFi-GAN |
| Frontend | React 18 (Vite), React Router, TanStack Query, Tailwind CSS, Framer Motion |

---

## Repository structure

```
NudiGuru/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app
│   │   ├── core/              # config, paths, security
│   │   ├── db/                # async engine + ORM models
│   │   ├── api/               # routers (auth, lessons, evaluate, tts, user)
│   │   ├── schemas/           # pydantic models
│   │   ├── data/              # lessons.py + generated templates/
│   │   ├── pipelines/         # dtw/ and hubert/ scoring, scoring.py
│   │   ├── tts/               # lazy TTS wrapper + engine
│   │   └── scripts/           # preprocess templates, convert voices
│   ├── alembic/               # database migrations
│   ├── tests/                 # pytest suite
│   ├── kn/                    # TTS weights (not committed)
│   ├── Voices/                # reference recordings (not committed)
│   ├── storage/               # uploads + TTS cache (not committed)
│   ├── main.py                # `uvicorn main:app` entry point
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── api/client.js      # single API client
│       ├── contexts/          # auth context
│       ├── components/ui/     # shared UI
│       └── pages/             # Dashboard, Practice, Lessons, Battle, Community, auth
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    ├── SETUP.md
    └── HACKATHON_GUIDE.md
```

---

## Quick start

Full instructions (including prerequisites and the TTS model): **[docs/SETUP.md](docs/SETUP.md)**.

### Backend

**Windows (CMD / Command Prompt)**

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
"C:\Program Files\PostgreSQL\18\bin\createdb.exe" -U postgres nudiguru
alembic upgrade head
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Windows (PowerShell)**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
Copy-Item .env.example .env
& "C:\Program Files\PostgreSQL\18\bin\createdb.exe" -U postgres nudiguru
alembic upgrade head
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Ubuntu (bash)**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres createdb nudiguru
alembic upgrade head
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bat
cd frontend
npm install
copy .env.example .env
npm run dev
```

The same commands work in PowerShell (use `Copy-Item` instead of `copy`) and Ubuntu (use `cp` instead of `copy`).

Open <http://localhost:5173> and sign up. API docs: <http://localhost:8000/docs>.

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — component map and scoring details
- [API reference](docs/API.md) — every endpoint with examples
- [Setup](docs/SETUP.md) — install, database, TTS model, voices, templates, tests

---

## Notes

- Battle mode is client-side; there is no server matchmaking.
- TTS weights, reference voices, and generated templates are not committed. See
  [Setup](docs/SETUP.md) to provide or regenerate them.
- The Community page is currently a UI prototype with local state only.

---

## Credits

- **IndicTTS** — Kannada speech data and baseline TTS models
- HuBERT, dtw, librosa, PyTorch & Transformers, FastAPI, React
