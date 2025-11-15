# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import os
import shutil
import io
import numpy as np
from scipy.io.wavfile import write as scipy_wav_write
import traceback
from typing import Dict
import uuid

# Import both pipelines
try:
    from WorkingPipeline.evaluate_speech import evaluate as evaluate_working
    from WorkingPipeline.syllables import WORD_MAP
    WORKING_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WorkingPipeline not available: {e}")
    WORKING_PIPELINE_AVAILABLE = False
    WORD_MAP = {}

try:
    from HubertPipeline.evaluate_speech import evaluate as evaluate_hubert
    HUBERT_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ HubertPipeline not available: {e}")
    HUBERT_PIPELINE_AVAILABLE = False

# ===========================
# TTS Setup
# ===========================
TTS_AVAILABLE = False
synthesizer = None
# ===========================
# TTS Setup - USING YOUR WORKING CODE
# ===========================
TTS_AVAILABLE = False
tts_engine = None
TTS_SAMPLE_RATE = 16000

try:
    from TTS_Module import generate_kannada_audio
    TTS_AVAILABLE = True
    print("✅ TTS Engine Loaded Successfully")
except ImportError as e:
    print(f"⚠️ TTS not available: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"⚠️ TTS initialization failed: {e}")
    traceback.print_exc()

app = FastAPI(title="NudiGuru API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length", "Content-Type"]
)

UPLOAD_DIR = "temp_uploads"
TTS_CACHE_DIR = "tts_cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# ===========================
# UTILITY FUNCTIONS
# ===========================

def normalize_audio(wav_array):
    """Normalize audio to prevent clipping"""
    if len(wav_array) == 0:
        return wav_array
    
    wav_array = np.array(wav_array, dtype=np.float32)
    wav_array = wav_array - np.mean(wav_array)
    
    max_val = np.max(np.abs(wav_array))
    if max_val > 0:
        wav_array = wav_array / max_val * 0.95
    
    return wav_array

# ===========================
# ENDPOINTS
# ===========================

@app.get("/")
def root():
    return {
        "status": "running",
        "working_pipeline": WORKING_PIPELINE_AVAILABLE,
        "hubert_pipeline": HUBERT_PIPELINE_AVAILABLE,
        "tts": TTS_AVAILABLE,
        "lessons": len(WORD_MAP)
    }

@app.get("/lessons")
def get_lessons():
    if not WORKING_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pipeline not available")
    
    lessons = []
    for word_id, data in WORD_MAP.items():
        lessons.append({
            "id": word_id,
            "order": int(word_id[1:]),
            "title": data["text"],
            "kannada_text": data["text"],
            "transliteration": data["text"],
            "english_translation": data["text"],
            "syllables": data["syllables"],
            "difficulty": "beginner" if int(word_id[1:]) <= 5 else "intermediate"
        })
    return sorted(lessons, key=lambda x: x["order"])

@app.post("/evaluate")
async def evaluate_pronunciation(
    audio: UploadFile = File(...),
    lesson_id: str = Form(...)
):
    print(f"🎯 Evaluating lesson: {lesson_id}")
    
    if lesson_id not in WORD_MAP:
        raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
    
    if not audio.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files accepted")
    
    temp_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"{lesson_id}_{audio.filename}"))
    expected = WORD_MAP[lesson_id]["text"]
    expected_audio_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"{lesson_id}_expected.wav"))

    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # ---------------------------------------
        # LIBROSA DISTANCE CHECK (First Priority)
        # ---------------------------------------
        try:
            import librosa
            import numpy as np
            from dtw import dtw

            def compare_audio(file1, file2, threshold=17500):
                """Compare two audio files using DTW distance"""
                # Load both audio files
                y1, sr1 = librosa.load(file1, sr=16000)
                y2, sr2 = librosa.load(file2, sr=16000)

                # Extract MFCC features
                mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
                mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

                # Run DTW
                dist, cost, acc, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: np.linalg.norm(x - y))

                print(f"DTW Distance: {dist}")

                # Decide similar or different
                if dist < threshold:
                    print("✅ The two spoken words are SIMILAR")
                    return dist, True
                elif dist < threshold + 3000:
                    print("⚠️ The two spoken words are SOMEWHAT SIMILAR")
                    return dist, True
                else:
                    print("❌ The two spoken words are DIFFERENT")
                    return dist, False

            # Generate expected audio if missing
            if not os.path.exists(expected_audio_path):
                print(f"🔊 Expected audio missing, generating: {expected_audio_path}")
                if TTS_AVAILABLE:
                    try:
                        from TTS_Module import generate_kannada_audio
                        audio_array, sample_rate = generate_kannada_audio(
                            text=expected,
                            speaker_name="female"
                        )
                        from scipy.io.wavfile import write as scipy_wav_write
                        scipy_wav_write(expected_audio_path, sample_rate, audio_array)
                        print(f"✅ Generated expected audio: {expected_audio_path}")
                    except Exception as gen_error:
                        print(f"⚠️ Could not generate expected audio: {gen_error}")
                else:
                    # Check cache directory
                    cache_path = os.path.join(TTS_CACHE_DIR, f"{lesson_id}.wav")
                    if os.path.exists(cache_path):
                        import shutil
                        shutil.copy(cache_path, expected_audio_path)
                        print(f"📋 Copied from cache: {cache_path} -> {expected_audio_path}")

            # Perform distance check
            if os.path.exists(expected_audio_path):
                distance, is_similar = compare_audio(temp_path, expected_audio_path)
                
                # If distance > 17500, reject immediately
                if not is_similar:
                    print(f"🚫 Distance check failed: {distance} > 17500")
                    return {
                        "accuracy_score": 0,
                        "syllables": [{"text": s, "accuracy": 0} for s in WORD_MAP[lesson_id]["syllables"]],
                        "areas_to_improve": [
                            f"Pronunciation doesn't match '{expected}'",
                            "The words are too different",
                            "Listen to reference and try again"
                        ],
                        "reference_audio_url": f"/tts/generate/{lesson_id}",
                        "stt_rejected": True,
                        "reason": "high_distance",
                        "distance": distance
                    }
                
                print(f"✅ Distance check passed: {distance} <= 4750")
            else:
                print(f"⚠️ Expected audio still not available, skipping distance check")
                
        except ImportError as e:
            print(f"⚠️ librosa or dtw not available: {e}")
        except Exception as e:
            print(f"⚠️ Distance check error (continuing): {e}")

        # ---------------------------------------
        # Continue with existing pipeline logic
        # ---------------------------------------
        results = {}

        # Run Working Pipeline
        if WORKING_PIPELINE_AVAILABLE:
            try:
                working_results = evaluate_working(temp_path, lesson_id)
                similarities = [r["similarity"] for r in working_results]
                working_accuracy = int(sum(similarities) / len(similarities) * 100)
                
                print(f"✅ Working Pipeline: {working_accuracy}%")
                
                results["working_pipeline"] = {
                    "accuracy": min(working_accuracy * 5, 100),  # Cap at 100
                    "syllables": [
                        {
                            "text": r["syllable"],
                            "accuracy": int(r["similarity"] * 100),
                            "distance": r.get("distance", 0)
                        }
                        for r in working_results
                    ]
                }
            except Exception as e:
                print(f"❌ Working Pipeline error: {e}")
                results["working_pipeline"] = {"error": str(e)}
        
        # Run HuBERT Pipeline
        if HUBERT_PIPELINE_AVAILABLE:
            try:
                hubert_results = evaluate_hubert(temp_path, lesson_id)
                similarities = [r["similarity"] for r in hubert_results]
                hubert_accuracy = int(sum(similarities) / len(similarities) * 100)
                
                print(f"✅ HuBERT Pipeline: {hubert_accuracy}%")
                
                results["hubert_pipeline"] = {
                    "accuracy": hubert_accuracy,
                    "syllables": [
                        {
                            "text": r["syllable"],
                            "accuracy": int(r["similarity"] * 100),
                            "correct": r.get("correct", False)
                        }
                        for r in hubert_results
                    ]
                }
            except Exception as e:
                print(f"❌ HuBERT Pipeline error: {e}")
                results["hubert_pipeline"] = {"error": str(e)}
        
        # Combine results
        if "working_pipeline" in results and "hubert_pipeline" in results:
            avg_accuracy = min(
                (results["working_pipeline"]["accuracy"] + results["hubert_pipeline"]["accuracy"]) // 2,
                100
            )
            
            combined_syllables = []
            for i, syl in enumerate(WORD_MAP[lesson_id]["syllables"]):
                wp_acc = results["working_pipeline"]["syllables"][i]["accuracy"]
                hp_acc = results["hubert_pipeline"]["syllables"][i]["accuracy"]
                
                combined_syllables.append({
                    "text": syl,
                    "accuracy": (wp_acc + hp_acc) // 2
                })
            
            results["combined"] = {
                "accuracy_score": avg_accuracy,
                "syllables": combined_syllables
            }
        elif "working_pipeline" in results:
            results["combined"] = {
                "accuracy_score": results["working_pipeline"]["accuracy"],
                "syllables": results["working_pipeline"]["syllables"]
            }
        elif "hubert_pipeline" in results:
            results["combined"] = {
                "accuracy_score": results["hubert_pipeline"]["accuracy"],
                "syllables": results["hubert_pipeline"]["syllables"]
            }
        else:
            raise HTTPException(status_code=503, detail="No pipeline available")
        
        # Generate improvement tips
        weak_syllables = [
            s for s in results["combined"]["syllables"] 
            if s["accuracy"] < 70
        ]
        
        tips = []
        if weak_syllables:
            tips = [f"Focus on '{s['text']}'" for s in weak_syllables[:3]]
        else:
            tips = ["Excellent pronunciation!"]
        
        return {
            "accuracy_score": results["combined"]["accuracy_score"],
            "syllables": results["combined"]["syllables"],
            "areas_to_improve": tips,
            "reference_audio_url": f"/tts/generate/{lesson_id}",
            "detailed_results": results
        }
    
    except Exception as e:
        print(f"❌ Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/tts/generate/{word_id}")
async def generate_tts_audio(word_id: str):
    """Generate TTS audio using YOUR working TTS engine"""
    if word_id not in WORD_MAP:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    kannada_text = WORD_MAP[word_id]["text"]
    cache_path = os.path.join(TTS_CACHE_DIR, f"{word_id}.wav")
    
    # Check cache first
    if os.path.exists(cache_path):
        print(f"✅ Serving cached TTS: {cache_path}")
        return FileResponse(
            cache_path,
            media_type="audio/wav",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Accept-Ranges": "bytes"
            }
        )
    
    # Check TTS availability
    if not TTS_AVAILABLE:
        print("❌ TTS not available")
        raise HTTPException(status_code=503, detail="TTS not available")
    
    try:
        from TTS_Module import generate_kannada_audio
        
        print(f"🎤 Generating TTS for: '{kannada_text}' (lesson {word_id})")
        
        # Generate audio using YOUR working function
        audio_array, sample_rate = generate_kannada_audio(
            text=kannada_text,
            speaker_name="female"  # or "male"
        )
        
        # Save to cache
        scipy_wav_write(cache_path, sample_rate, audio_array)
        print(f"💾 Cached to: {cache_path}")
        
        # Return file
        return FileResponse(
            cache_path,
            media_type="audio/wav",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Accept-Ranges": "bytes"
            }
        )
    
    except Exception as e:
        print(f"❌ TTS Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    
@app.get("/user/stats")
def get_user_stats():
    return {
        "full_name": "NudiGuru User",
        "email": "user@nudiguru.com",
        "daily_practice_count": 5,
        "daily_goal": 10,
        "streak_days": 7,
        "total_practices": 42
    }

@app.get("/tts/status")
def tts_status():
    status = {
        "available": TTS_AVAILABLE,
        "synthesizer_loaded": synthesizer is not None,
        "cache_dir": TTS_CACHE_DIR,
        "cached_files": len(os.listdir(TTS_CACHE_DIR)) if os.path.exists(TTS_CACHE_DIR) else 0
    }
    
    if synthesizer and hasattr(synthesizer.tts_model, 'speaker_manager'):
        if synthesizer.tts_model.speaker_manager:
            status["speakers"] = synthesizer.tts_model.speaker_manager.speaker_names
        else:
            status["speakers"] = ["single_speaker_model"]
    
    return status

# Battle room management
battle_rooms: Dict[str, dict] = {}

# ===========================
# BATTLE ENDPOINTS
# ===========================

@app.post("/battle/create")
async def create_battle_room():
    """Create a new battle room"""
    room_code = str(uuid.uuid4())[:6].upper()
    
    # Random lesson selection
    lesson_ids = list(WORD_MAP.keys())
    selected_lesson = lesson_ids[0]  # You can randomize this
    
    battle_rooms[room_code] = {
        "lesson_id": selected_lesson,
        "lesson_text": WORD_MAP[selected_lesson]["text"],
        "players": {},
        "status": "waiting"
    }
    
    print(f"🎮 Created battle room: {room_code}")
    
    return {
        "room_code": room_code,
        "lesson_id": selected_lesson,
        "lesson_text": WORD_MAP[selected_lesson]["text"]
    }

@app.get("/battle/room/{room_code}")
async def get_battle_room(room_code: str):
    """Get battle room info"""
    if room_code not in battle_rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return battle_rooms[room_code]

@app.post("/battle/score")
async def score_battle_audio(
    audio: UploadFile = File(...),
    room_code: str = Form(...),
    player_id: str = Form(...)
):
    """Score a player's pronunciation in battle"""
    
    if room_code not in battle_rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    room = battle_rooms[room_code]
    lesson_id = room["lesson_id"]
    
    print(f"⚔️ Scoring battle for room {room_code}, player {player_id}")
    
    # Save audio temporarily
    temp_path = os.path.join(UPLOAD_DIR, f"battle_{room_code}_{player_id}.wav")
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        # Use existing evaluation pipeline
        results = {}
        
        # Working Pipeline
        if WORKING_PIPELINE_AVAILABLE:
            try:
                working_results = evaluate_working(temp_path, lesson_id)
                similarities = [r["similarity"] for r in working_results]
                working_accuracy = int(sum(similarities) / len(similarities) * 100)
                results["working"] = working_accuracy
            except Exception as e:
                print(f"❌ Working pipeline error: {e}")
                results["working"] = 0
        
        # HuBERT Pipeline
        if HUBERT_PIPELINE_AVAILABLE:
            try:
                hubert_results = evaluate_hubert(temp_path, lesson_id)
                similarities = [r["similarity"] for r in hubert_results]
                hubert_accuracy = int(sum(similarities) / len(similarities) * 100)
                results["hubert"] = hubert_accuracy
            except Exception as e:
                print(f"❌ HuBERT pipeline error: {e}")
                results["hubert"] = 0
        
        # Combined score
        if "working" in results and "hubert" in results:
            final_score = (results["working"] + results["hubert"]) // 2
        elif "working" in results:
            final_score = results["working"]
        elif "hubert" in results:
            final_score = results["hubert"]
        else:
            final_score = 0
        
        # Store in room
        room["players"][player_id] = {
            "score": final_score,
            "scored_at": None  # Add timestamp if needed
        }
        
        print(f"✅ Player {player_id} scored: {final_score}%")
        
        # Check if both players scored
        both_scored = len(room["players"]) == 2
        winner = None
        
        if both_scored:
            players = list(room["players"].items())
            if players[0][1]["score"] > players[1][1]["score"]:
                winner = players[0][0]
            elif players[1][1]["score"] > players[0][1]["score"]:
                winner = players[1][0]
            else:
                winner = "tie"
        
        return {
            "score": final_score,
            "player_id": player_id,
            "room_status": "complete" if both_scored else "waiting",
            "winner": winner,
            "all_scores": room["players"]
        }
    
    except Exception as e:
        print(f"❌ Error: {traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)