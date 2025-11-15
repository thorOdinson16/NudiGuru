# NudiGuru – AI-Powered Kannada Pronunciation Learning Platform

NudiGuru is an AI-powered Kannada pronunciation learning platform that helps users speak Kannada accurately through real-time feedback, TTS audio guidance, and syllable-level scoring.  
It brings the experience of a smart language coach — designed specifically for Indian languages and powered by modern deep-learning pipelines.

---

## 🚀 Features

### 🎤 Pronunciation Evaluation (Dual Pipeline)
NudiGuru uses a **hybrid evaluation system** with two independent pipelines:

#### **WorkingPipeline (DTW-based)**
- Uses **Log-Mel Spectrogram** features (40 mel bins)
- **Pre-emphasis filtering** to boost high frequencies
- **CMVN normalization** for consistent feature representation
- **DTW (Dynamic Time Warping)** distance calculation
- Fixed threshold-based scoring (threshold: 700)
- Syllable-level distance, similarity, and correctness

#### **HuBERT Pipeline (Deep Learning-based)**
- Uses **facebook/hubert-base-ls960** model from HuggingFace
- Extracts **768-dimensional embeddings** from audio
- **Cosine similarity** comparison with reference templates
- Threshold-based scoring (threshold: 0.70)
- More robust to variations in speaking speed and style

Both pipelines provide:
- Syllable-level similarity scores
- Acoustic feature comparison
- Binary correctness flags
- Combined accuracy metrics

The system generates a **combined accuracy score** and highlights **weak syllables** for targeted improvement.

---

### 🔊 Kannada Text-to-Speech (TTS)
NudiGuru supports high-quality Kannada TTS using:

- **FastPitch** (acoustic model)  
- **HiFi-GAN V1** (vocoder)  

TTS is used to:
- Generate reference pronunciation
- Provide replay guidance to learners
- Produce lesson audio automatically

The model (≈1.5 GB) is downloaded via **GitHub Releases**.

---

### 📚 Lessons System
Each lesson contains:
- A Kannada word (stored in `WORD_MAP`)
- Syllable breakdown  
- Expected pronunciation  
- TTS reference audio  
- Difficulty level  

Lessons are served directly from `WORD_MAP` in the backend (`syllables.py`).

---

### ⚔️ Pronunciation Battle Mode
Users can join a room and compete on:
- Accuracy score  
- Speed  
- Correct syllables  

The backend generates a room code, processes both pronunciations using the dual-pipeline system, and determines a winner.

---

### 🌐 Tech Stack

#### Backend
- **FastAPI** (REST API framework)
- **Python 3.10+**  
- **Librosa** (audio processing)
- **NumPy** (numerical computing)
- **DTW** (Dynamic Time Warping)
- **HuBERT** (PyTorch + Transformers)
- **FastPitch + HiFi-GAN** (IndicTTS-based TTS model)
- **Pydub** (audio manipulation)

#### Frontend
- **React (Vite)**  
- JSX components with custom UI  
- Pages: Practice, Lessons, Community, Battle Mode, etc.

---

## 🗂 Folder Structure

```
NudiGuru/
│
├── backend/
│   ├── HubertPipeline/
│   │   ├── evaluate_speech.py      # HuBERT evaluation entry point
│   │   ├── features_hubert.py      # HuBERT embedding extraction
│   │   ├── scorer_hubert.py        # Cosine similarity scoring
│   │   ├── preprocess_references.py # Generate HuBERT templates
│   │   ├── syllables.py            # Word map with syllables
│   │   └── syllable_templates.json # Pre-computed HuBERT embeddings
│   │
│   ├── WorkingPipeline/
│   │   ├── evaluate_speech.py      # DTW evaluation entry point
│   │   ├── features.py             # Log-Mel feature extraction
│   │   ├── mel_dtw.py              # DTW scoring logic
│   │   ├── preprocess_references.py # Generate DTW templates
│   │   ├── syllables.py            # Word map with syllables
│   │   └── syllable_templates.json # Pre-computed Mel features
│   │
│   ├── temp_uploads/               # Temporary user audio uploads
│   ├── tts_cache/                  # Cached TTS audio files
│   ├── kn/                         # TTS model folder (after download)
│   ├── Voices/                     # Reference audio samples
│   │   ├── Speaker1/
│   │   │   ├── 1.wav
│   │   │   ├── 2.wav
│   │   │   └── ...
│   │   ├── Speaker2/
│   │   └── ...
│   │
│   ├── main.py                     # FastAPI backend entrypoint
│   ├── TTS_Module.py               # TTS generation logic
│   └── requirements.txt
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

The release contains the Kannada FastPitch + HiFi-GAN model.

### 2️⃣ Extract into backend folder

Place the extracted model here:

```
NudiGuru/backend/kn/
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

**Download the TTS model** (see section above)

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

## 🎙️ Adding Custom Reference Voices

If you want to add additional reference speakers or update the pronunciation templates:

### 1️⃣ Add Audio Files
Place speaker audio files in the `Voices/` directory:
```
backend/Voices/
├── Speaker1/
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
├── Speaker2/
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
└── YourNewSpeaker/
    ├── 1.wav
    ├── 2.wav
    └── ...
```

**Important:**
- Each speaker should have their own folder
- Audio files must be named by word ID (e.g., `1.wav`, `2.wav`, etc.)
- Audio must be in WAV format
- Word IDs correspond to entries in `WORD_MAP`

### 2️⃣ Regenerate Templates for WorkingPipeline
```bash
cd backend/WorkingPipeline
python preprocess_references.py
```

This will:
- Process all speaker audio files
- Extract Log-Mel features for each syllable
- Generate `syllable_templates.json` with DTW templates

### 3️⃣ Regenerate Templates for HuBERT Pipeline
```bash
cd backend/HubertPipeline
python preprocess_references.py
```

This will:
- Process all speaker audio files
- Extract HuBERT embeddings for each syllable
- Generate `syllable_templates.json` with embedding templates

**Note:** HuBERT preprocessing requires downloading the model (~360MB) and takes longer to complete.

### 4️⃣ Restart the Server
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The new templates will be loaded automatically.

---

## 🔌 API Overview (Backend)

### `GET /`
Health check + loaded module info.

### `GET /lessons`
Returns all lessons with:
- Kannada text
- Syllables
- Ordering
- Difficulty

### `POST /evaluate`
Upload a WAV file and get pronunciation feedback.

### `GET /tts/generate/{lesson_id}`
Generates (or retrieves cached) TTS audio for the lesson.

### Battle Mode Endpoints

#### `POST /battle/upload`
Upload pronunciation for battle scoring.

#### `POST /battle/score`
Calculate battle results and determine winner.

---

## 🧠 How the Pronunciation Scoring Works

### WorkingPipeline (DTW-based)

1. **Audio Preprocessing**
   - Load audio at 16kHz sample rate
   - Apply pre-emphasis filter (α = 0.97) to boost high frequencies
   - Trim silence (25dB threshold)

2. **Feature Extraction**
   - Compute Log-Mel Spectrogram (40 mel bins)
   - Window: 25ms (400 samples)
   - Hop: 10ms (160 samples)
   - Frequency range: 20Hz - 7600Hz

3. **Normalization**
   - Apply Cepstral Mean and Variance Normalization (CMVN)

4. **DTW Scoring**
   - Compare user features with reference templates
   - Calculate DTW distance using L2 norm
   - Take minimum distance across all reference speakers
   - Threshold: 700 (distances below this are considered correct)

5. **Similarity Score**
   - Convert to similarity: `1.0 - min(distance/threshold, 1.0)`
   - Range: 0.0 to 1.0

### HuBERT Pipeline (Deep Learning)

1. **Audio Preprocessing**
   - Load audio at 16kHz sample rate
   - Trim silence
   - Handle short audio with zero-padding

2. **Feature Extraction**
   - Use facebook/hubert-base-ls960 model
   - Extract 768-dimensional embeddings
   - Average pooling over time dimension
   - L2 normalization

3. **Cosine Similarity**
   - Compare user embedding with reference embeddings
   - Calculate cosine similarity
   - Take maximum similarity across all reference speakers
   - Threshold: 0.70 (similarities above this are considered correct)

4. **Similarity Score**
   - Direct cosine similarity value
   - Range: -1.0 to 1.0 (typically 0.0 to 1.0)

### Combined Scoring

Both pipelines run independently and provide:
- Syllable-level accuracy
- Overall word accuracy
- Binary correctness flags
- Similarity metrics

The frontend can display results from either or both pipelines for comparison.

---

## 📜 Credits & Acknowledgements

### IndicTTS
The Kannada speech data and baseline models originate from the IndicTTS initiative.
These assets were instrumental for building the FastPitch + HiFi-GAN Kannada TTS model.

### Other Tools & Libraries
- **HuBERT**
- **Whisper** 
- **DTW-Python** 
- **Librosa**
- **PyTorch & Transformers**
- **FastAPI**
- **React**

---

## 🐛 Troubleshooting

### TTS Model Not Found
- Ensure you've downloaded the model from GitHub Releases
- Check that `backend/kn/` contains all required files
- Restart the FastAPI server

### Low Accuracy Scores
- Ensure audio is clear with minimal background noise
- Speak at a normal pace
- Position microphone properly
- Check that reference voices are high quality

### Template Generation Fails
- Ensure `Voices/` directory structure is correct
- Check that all audio files are valid WAV format
- Verify file naming matches word IDs in `WORD_MAP`
- For HuBERT: Ensure sufficient RAM (4GB+ recommended)

### Server Crashes
- Check Python version (3.10+ required)
- Verify all dependencies are installed
- Check for port conflicts (8000 for backend, 5173 for frontend)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description
