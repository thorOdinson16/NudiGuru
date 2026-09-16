# NudiGuru Setup

Instructions are given separately for **Windows** and **Ubuntu**. Run all
backend commands from the `backend/` directory and all frontend commands from
the `frontend/` directory.

## Prerequisites

| Requirement | Windows | Ubuntu |
| --- | --- | --- |
| Python 3.10+ | Install from <https://www.python.org/downloads/> (tick "Add python.exe to PATH") | `sudo apt update && sudo apt install -y python3 python3-venv python3-pip` |
| Node.js 18+ | Install from <https://nodejs.org/> or `winget install OpenJS.NodeJS.LTS` | `sudo apt install -y nodejs npm` |
| PostgreSQL 13+ | Install from <https://www.postgresql.org/download/windows/> | `sudo apt install -y postgresql` |
| ffmpeg | `winget install Gyan.FFmpeg` (or from <https://ffmpeg.org/>) and ensure it is on PATH | `sudo apt install -y ffmpeg` |

Verify: `python --version`, `node --version`, `psql --version`, `ffmpeg -version`.

---

## Backend

### 1. Create and activate a virtual environment

**Windows (CMD / Command Prompt)**

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked, allow it for this session first:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Ubuntu (bash)**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.lock.txt` captures the exact versions of a fully verified working
environment (including the ML stack). Use it if you need a reproducible install:

```bash
pip install -r requirements.lock.txt
```

If you have a CUDA GPU, install the matching PyTorch build for faster
HuBERT/TTS inference and set `TTS_DEVICE=cuda` in `.env`.

### 3. Configure environment

**Windows (CMD)**

```bat
copy .env.example .env
```

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

**Ubuntu**

```bash
cp .env.example .env
```

Edit `.env` and set `DATABASE_URL`, `JWT_SECRET`, and `CORS_ORIGINS`.
`DATABASE_URL` ships with a placeholder password — replace
`YOUR_POSTGRES_PASSWORD` with your PostgreSQL password, for example:
`postgresql+asyncpg://postgres:mysecret@localhost:5432/nudiguru`.

### 4. Create the database and run migrations

**Windows (CMD or PowerShell)**

```bat
:: CMD (enter the postgres password when prompted)
"C:\Program Files\PostgreSQL\18\bin\createdb.exe" -U postgres nudiguru
```

```powershell
# PowerShell
& "C:\Program Files\PostgreSQL\18\bin\createdb.exe" -U postgres nudiguru
```

Then set `DATABASE_URL` in `.env` to match that password, e.g.
`postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/nudiguru`, and run:

```bat
alembic upgrade head
```

**Ubuntu**

```bash
# Give the postgres role a password and create the database
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres createdb nudiguru

alembic upgrade head
```

> If you use a password other than `postgres`, update `DATABASE_URL` in `.env`.

### 5. Provide the TTS model (~1.6 GB)

The Kannada FastPitch + HiFi-GAN weights are not committed. Download them from
the GitHub Releases page and extract so the layout is:

```
backend/kn/fastpitch/best_model.pth
backend/kn/fastpitch/config.json
backend/kn/fastpitch/speakers.pth
backend/kn/hifigan/best_model.pth
backend/kn/hifigan/config.json
```

Without them the API still starts and reports `tts_model: false`; the `/tts/*`
endpoints return `503`.

### 6. Provide reference voices and templates

See [Reference voices](#reference-voices) below. The scoring pipelines need the
generated templates in `backend/app/data/templates/`.

### 7. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Windows and Ubuntu use the same command. Health check:
<http://localhost:8000/health>. Interactive API docs:
<http://localhost:8000/docs>.

---

## Frontend

From the `frontend/` directory:

**Install**

```bash
npm install
```

**Configure**

```bat
copy .env.example .env        # Windows CMD
```

```powershell
Copy-Item .env.example .env   # Windows PowerShell
```

```bash
cp .env.example .env          # Ubuntu
```

Set `VITE_API_URL` if the backend is not on `http://localhost:8000`.

**Run**

```bash
npm run dev
```

The app runs at <http://localhost:5173> (same on both platforms). Register an
account on the Sign Up page — authentication is required to access app routes.

---

## Reference voices

Reference recordings live under `backend/Voices/<Speaker>/<wordId>.wav`, e.g.
`backend/Voices/PriyaF/w01.wav` corresponds to lesson `w01`. Each speaker folder
should contain one WAV per lesson id (`w01.wav` … `w15.wav`). The eight bundled
speakers are committed to the repository.

To convert source recordings (e.g. `.ogg`) into the expected numbered WAVs
(run from `backend/`, with the venv active):

```bash
python -m app.scripts.convert_voices --folder "path/to/speaker" --ext .ogg
```

Then regenerate the syllable templates:

```bash
python -m app.scripts.preprocess_dtw
python -m app.scripts.preprocess_hubert
```

These write `backend/app/data/templates/dtw.json` and `hubert.json`
(not committed). HuBERT preprocessing downloads `facebook/hubert-base-ls960`
(~360 MB) on first run and needs 4 GB+ RAM.

---

## Tests

With the venv active, from `backend/`:

```bash
pytest
```

`tests/test_api.py` runs the FastAPI app via the test client; the DB-dependent
endpoints are not exercised there, so a running PostgreSQL is not required for
the current suite.

---

## Troubleshooting

- **`503` from `/evaluate`** — templates missing; run the preprocess scripts.
- **`503` from `/tts/*`** — TTS weights missing or `TTS_DEVICE` misconfigured.
- **`401` on `/user/*`** — missing/expired token; sign in again.
- **`InvalidPasswordError: password authentication failed for user "postgres"`**
  — the password in `.env`'s `DATABASE_URL` does not match your PostgreSQL
  password. Update it (replace `YOUR_POSTGRES_PASSWORD`) and re-run
  `alembic upgrade head`.
- **`502`/connection errors from the frontend** — check `VITE_API_URL` and that
  the backend is running; CORS only allows origins in `CORS_ORIGINS`.
- **Audio slicing errors** — install `ffmpeg` and ensure it is on PATH.
- **Cannot activate the venv in cmd** — in Command Prompt use
  `.venv\Scripts\activate.bat` (or just `.venv\Scripts\activate`). The
  `Activate.ps1` script only works in PowerShell; using it in cmd fails.
- **Activation blocked in PowerShell** — run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate.
- **"activate is not recognized"** — make sure you are inside the `backend`
  directory (the path is relative), or call it with the full path.
- **`createdb`/`psql` not found (Windows)** — use the full path under
  `C:\Program Files\PostgreSQL\<version>\bin\` or add it to PATH.
