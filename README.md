# NudiGuru – AI-Powered Kannada Pronunciation Learning Platform

NudiGuru is an AI-powered Kannada pronunciation learning platform that helps users speak Kannada accurately through real-time feedback, TTS audio guidance, and syllable-level scoring.  
It brings the experience of a smart language coach — designed specifically for Indian languages and powered by modern deep-learning pipelines.

---

## 🚀 Features

### 🎤 Pronunciation Evaluation (Dual Pipeline)
NudiGuru uses a hybrid evaluation system:

- **WorkingPipeline**  
- **HuBERT-based Pipeline**

Both provide syllable-level similarity scores using:
- MFCC + DTW distance
- Phoneme alignment
- Acoustic-feature comparison

The system generates a **combined accuracy score** and highlights **weak syllables** to improve.

---

### 🔊 Kannada Text-to-Speech (TTS)
NudiGuru supports high-quality Kannada TTS using:

- **FastPitch (acoustic model)**  
- **HiFi-GAN V1 (vocoder)**  

TTS is used to:
- Generate reference pronunciation
- Provide replay guidance to learners
- Produce lesson audio automatically

The model (≈1.5 GB) is downloaded via **GitHub Releases**.

---

### 📚 Lessons System
Each lesson contains:
- A Kannada word  
- Syllable breakdown  
- Expected pronunciation  
- TTS reference audio  
- Difficulty level  

Lessons are served directly from `WORD_MAP` in the backend.

---

### ⚔️ Pronunciation Battle Mode
Users can join a room and compete on:
- Accuracy score  
- Speed  
- Correct syllables  

The backend generates a room code, processes both pronunciations, and determines a winner.

---

### 🌐 Tech Stack

#### Backend
- **FastAPI**  
- **Python**  
- **Librosa, NumPy, DTW**  
- **HuBERT** (PyTorch)  
- **FastPitch + HiFiGAN** (IndicTTS-based TTS model)  

#### Frontend
- **React (Vite)**  
- JSX components with custom UI  
- Pages like Practice, Lessons, Community, Battle Mode, etc.

---

## 🗂 Folder Structure

```
NudiGuru/
│
├── backend/
│   ├── HubertPipeline/
│   ├── WorkingPipeline/
│   ├── temp_uploads/
│   ├── tts_cache/
│   ├── kn/                  # TTS model folder (after download)
│   ├── Voices/
│   ├── main.py              # FastAPI backend entrypoint
│   └── TTS_Module.py
│
└── frontend/
    ├── api/
    ├── components/
    ├── pages/
    ├── main.jsx
    ├── App.jsx
    └── index.css
```

---

## 📥 Installing the Kannada TTS Model

The TTS model (~1.5 GB) **must be downloaded manually**.

### 1️⃣ Download from GitHub Releases

Download the ZIP file:

👉 **https://github.com/thorOdinson16/NudiGuru/releases/latest**

The release contains the Kannada FastPitch + HiFiGAN model.

### 2️⃣ Extract into backend folder

Place the extracted model here:

```
NudiGuru/backend/kn/
```

The folder should contain:

```
config.json
model.pth
hifigan_generator.pth
hifigan_config.json
```

Your backend will automatically load the model on startup.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/NudiGuru.git
cd NudiGuru
```

### 2. Setup Backend (FastAPI)

**Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```
(Ensure Python 3.10+)

**Run FastAPI server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend now runs at:
```
http://localhost:8000
```

### 3. Setup Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:
```
http://localhost:5173
```

---

## 🔌 API Overview (Backend)

### `GET /`
Health check + loaded module info.

### `GET /lessons`
Returns all lessons with:
- Kannada text
- syllables
- ordering
- difficulty

### `POST /evaluate`
Upload a WAV file and get:
- Accuracy score
- Syllable-level accuracy
- Areas to improve
- DTW distance
- Reference TTS URL

Used by Practice & Battle mode.

### `GET /tts/generate/{lesson_id}`
Generates (or retrieves cached) TTS audio for the lesson.

### `/battle/*` endpoints
- Create battle
- Join room
- Upload audio
- Score battle
- Determine winner

---

## 🧠 How the Pronunciation Scoring Works

### 1️⃣ DTW-based audio distance
MFCC features → DTW alignment → similarity estimation.

### 2️⃣ WorkingPipeline
- Syllable segmentation
- Feature comparison
- Relative similarity score

### 3️⃣ HuBERT-based Deep Model
- Embedding similarity
- Phoneme-based scoring

### 4️⃣ Combined accuracy score
A weighted average produces final accuracy.

---

## 📜 Credits & Acknowledgements

### IndicTTS
The Kannada speech data and baseline models originate from the IndicTTS initiative under:
- IIT Madras
- C-DAC
- TDIL Programme, MeitY (Govt. of India)

These assets were instrumental for building the FastPitch + HiFiGAN Kannada TTS model.

### Other Tools
- Whisper (OpenAI)
- DTW
- Librosa
- FastAPI
- React
