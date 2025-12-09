import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Trophy, Volume2, RotateCcw, Loader2 } from "lucide-react";

const API_URL = "http://localhost:8000";

/* ---------------------------------------------------------
   WAVEFORM VISUALIZER
---------------------------------------------------------- */
function Wave({ active }) {
  return (
    <div className="flex gap-[3px] h-20 items-end justify-center">
      {Array.from({ length: 40 }).map((_, i) => (
        <motion.div
          key={i}
          animate={{
            height: active ? Math.random() * 70 + 10 : 10,
          }}
          className={`w-1 rounded-full ${
            active
              ? "bg-gradient-to-t from-orange-500 to-pink-500"
              : "bg-gray-300"
          }`}
          transition={{ duration: 0.1 }}
        />
      ))}
    </div>
  );
}

/* ---------------------------------------------------------
   HELPER: Audio Buffer to WAV
---------------------------------------------------------- */
function audioBufferToWav(audioBuffer) {
  const numberOfChannels = 1;
  const sampleRate = 16000;
  const length = audioBuffer.length;
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, length * 2, true);

  const channelData = audioBuffer.getChannelData(0);
  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/* ---------------------------------------------------------
   API FUNCTIONS
---------------------------------------------------------- */
async function fetchRandomLesson() {
  const response = await fetch(`${API_URL}/lessons`);
  if (!response.ok) throw new Error("Failed to fetch lessons");
  const lessons = await response.json();
  return lessons[Math.floor(Math.random() * lessons.length)];
}

async function evaluatePronunciation(audioBlob, lessonId) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");
  formData.append("lesson_id", lessonId);

  const response = await fetch(`${API_URL}/evaluate`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) throw new Error("Evaluation failed");
  return response.json();
}

/* ---------------------------------------------------------
   MAIN COMPONENT
---------------------------------------------------------- */
export default function PronounciationBattlePage() {
  const [step, setStep] = useState("setup"); // setup | player1 | player2 | results
  const [lesson, setLesson] = useState(null);
  const [player1Name, setPlayer1Name] = useState("");
  const [player2Name, setPlayer2Name] = useState("");
  const [recording, setRecording] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [player1Score, setPlayer1Score] = useState(null);
  const [player2Score, setPlayer2Score] = useState(null);
  const [isPlayingTTS, setIsPlayingTTS] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);

  /* -------------------- SETUP -------------------- */
  const startBattle = async () => {
    if (!player1Name || !player2Name) {
      alert("Please enter both player names!");
      return;
    }

    try {
      const randomLesson = await fetchRandomLesson();
      setLesson(randomLesson);
      setStep("player1");
    } catch (error) {
      alert("Failed to load lesson. Please try again.");
    }
  };

  /* -------------------- PLAY TTS -------------------- */
  const playTTS = async () => {
    if (!lesson) return;

    try {
      setIsPlayingTTS(true);
      const response = await fetch(`${API_URL}/tts/generate/${lesson.id}`);
      if (!response.ok) throw new Error("TTS failed");

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        setIsPlayingTTS(false);
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = () => {
        setIsPlayingTTS(false);
        alert("Failed to play audio");
      };

      await audio.play();
    } catch (error) {
      console.error("TTS error:", error);
      setIsPlayingTTS(false);
      alert("Could not play reference audio");
    }
  };

  /* -------------------- RECORDING -------------------- */
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext ||
        window.webkitAudioContext)();

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) =>
        audioChunksRef.current.push(e.data);

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        try {
          const arrayBuffer = await audioBlob.arrayBuffer();
          const audioBuffer = await audioContextRef.current.decodeAudioData(
            arrayBuffer
          );
          const wavBlob = audioBufferToWav(audioBuffer);
          await handleEvaluation(wavBlob);
        } catch (err) {
          console.error("Audio processing failed:", err);
          alert("Failed to process audio. Please try again.");
        }
      };

      mediaRecorderRef.current.start();
      setRecording(true);
    } catch (error) {
      console.error("Microphone error:", error);
      alert("Please allow microphone access");
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    setRecording(false);
  };

  /* -------------------- EVALUATION -------------------- */
  const handleEvaluation = async (wavBlob) => {
    setEvaluating(true);

    try {
      const result = await evaluatePronunciation(wavBlob, lesson.id);

      if (step === "player1") {
        setPlayer1Score(result.accuracy_score);
        setStep("player2");
      } else if (step === "player2") {
        setPlayer2Score(result.accuracy_score);
        setStep("results");
      }
    } catch (error) {
      console.error("Evaluation failed:", error);
      alert("Failed to evaluate. Please try again.");
    } finally {
      setEvaluating(false);
    }
  };

  /* -------------------- RESET -------------------- */
  const resetBattle = () => {
    setStep("setup");
    setLesson(null);
    setPlayer1Name("");
    setPlayer2Name("");
    setPlayer1Score(null);
    setPlayer2Score(null);
  };

  /* ========================================================= */

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* TITLE */}
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-gray-900 mb-8 flex items-center gap-3"
        >
          <Trophy className="text-orange-600 w-10 h-10" />
          Pronunciation Battle
        </motion.h1>

        {/* ---------------------------------------------------------
            STEP 1: SETUP
        ---------------------------------------------------------- */}
        {step === "setup" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-md mx-auto"
          >
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h2 className="text-2xl font-bold mb-6 text-center">
                Setup Battle
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Player 1 Name
                  </label>
                  <input
                    type="text"
                    value={player1Name}
                    onChange={(e) => setPlayer1Name(e.target.value)}
                    placeholder="Enter name"
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Player 2 Name
                  </label>
                  <input
                    type="text"
                    value={player2Name}
                    onChange={(e) => setPlayer2Name(e.target.value)}
                    placeholder="Enter name"
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  />
                </div>

                <button
                  onClick={startBattle}
                  className="w-full mt-6 px-6 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white text-lg font-bold rounded-xl hover:shadow-xl transition"
                >
                  Start Battle! ⚔️
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* ---------------------------------------------------------
            STEP 2: PLAYER 1 RECORDING
        ---------------------------------------------------------- */}
        {step === "player1" && lesson && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-2xl mx-auto"
          >
            <div className="bg-gradient-to-br from-orange-50 to-pink-50 rounded-2xl shadow-xl p-8">
              <h2 className="text-3xl font-bold mb-2 text-center text-orange-600">
                {player1Name}'s Turn
              </h2>
              <p className="text-center text-gray-600 mb-6">
                Listen and repeat the phrase
              </p>

              {/* Lesson Display */}
              <div className="bg-white rounded-xl p-6 mb-6 text-center shadow-md">
                <p className="text-4xl font-bold kannada-text mb-2">
                  {lesson.kannada_text}
                </p>
                <p className="text-lg text-gray-700 italic mb-1">
                  {lesson.transliteration}
                </p>
                <p className="text-sm text-gray-600">
                  "{lesson.english_translation}"
                </p>
              </div>

              {/* Play TTS */}
              <div className="flex justify-center mb-6">
                <button
                  onClick={playTTS}
                  disabled={recording || isPlayingTTS || evaluating}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:bg-gray-300 transition"
                >
                  <Volume2 className="w-5 h-5" />
                  {isPlayingTTS ? "Playing..." : "Hear Pronunciation"}
                </button>
              </div>

              {/* Waveform */}
              <div className="mb-6">
                <Wave active={recording} />
              </div>

              {/* Record Button */}
              <div className="flex justify-center">
                {evaluating ? (
                  <div className="flex items-center gap-3 text-orange-600">
                    <Loader2 className="w-8 h-8 animate-spin" />
                    <span className="text-lg font-medium">Evaluating...</span>
                  </div>
                ) : (
                  <motion.button
                    whileTap={{ scale: 0.95 }}
                    onClick={recording ? stopRecording : startRecording}
                    disabled={evaluating}
                    className={`relative w-32 h-32 rounded-full flex items-center justify-center shadow-2xl transition ${
                      recording
                        ? "bg-gradient-to-br from-red-500 to-pink-600"
                        : "bg-gradient-to-br from-orange-500 to-pink-500 hover:shadow-xl"
                    }`}
                  >
                    {recording && (
                      <motion.div
                        animate={{ scale: [1, 1.3, 1] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                        className="absolute inset-0 rounded-full bg-red-500 opacity-20"
                      />
                    )}
                    <Mic className="w-16 h-16 text-white" />
                  </motion.button>
                )}
              </div>

              <p className="text-center text-gray-600 mt-4">
                {recording
                  ? "Recording... Tap to stop"
                  : evaluating
                  ? "Processing your pronunciation..."
                  : "Tap to record"}
              </p>
            </div>
          </motion.div>
        )}

        {/* ---------------------------------------------------------
            STEP 3: PLAYER 2 RECORDING
        ---------------------------------------------------------- */}
        {step === "player2" && lesson && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-2xl mx-auto"
          >
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl shadow-xl p-8">
              <h2 className="text-3xl font-bold mb-2 text-center text-indigo-600">
                {player2Name}'s Turn
              </h2>
              <p className="text-center text-gray-600 mb-6">
                Listen and repeat the phrase
              </p>

              {/* Show Player 1's Score */}
              <div className="bg-white rounded-xl p-4 mb-6 text-center shadow-md">
                <p className="text-sm text-gray-600">
                  {player1Name}'s Score:
                </p>
                <p className="text-3xl font-bold text-orange-600">
                  {player1Score}%
                </p>
              </div>

              {/* Lesson Display */}
              <div className="bg-white rounded-xl p-6 mb-6 text-center shadow-md">
                <p className="text-4xl font-bold kannada-text mb-2">
                  {lesson.kannada_text}
                </p>
                <p className="text-lg text-gray-700 italic mb-1">
                  {lesson.transliteration}
                </p>
                <p className="text-sm text-gray-600">
                  "{lesson.english_translation}"
                </p>
              </div>

              {/* Play TTS */}
              <div className="flex justify-center mb-6">
                <button
                  onClick={playTTS}
                  disabled={recording || isPlayingTTS || evaluating}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:bg-gray-300 transition"
                >
                  <Volume2 className="w-5 h-5" />
                  {isPlayingTTS ? "Playing..." : "Hear Pronunciation"}
                </button>
              </div>

              {/* Waveform */}
              <div className="mb-6">
                <Wave active={recording} />
              </div>

              {/* Record Button */}
              <div className="flex justify-center">
                {evaluating ? (
                  <div className="flex items-center gap-3 text-indigo-600">
                    <Loader2 className="w-8 h-8 animate-spin" />
                    <span className="text-lg font-medium">Evaluating...</span>
                  </div>
                ) : (
                  <motion.button
                    whileTap={{ scale: 0.95 }}
                    onClick={recording ? stopRecording : startRecording}
                    disabled={evaluating}
                    className={`relative w-32 h-32 rounded-full flex items-center justify-center shadow-2xl transition ${
                      recording
                        ? "bg-gradient-to-br from-red-500 to-pink-600"
                        : "bg-gradient-to-br from-blue-500 to-indigo-500 hover:shadow-xl"
                    }`}
                  >
                    {recording && (
                      <motion.div
                        animate={{ scale: [1, 1.3, 1] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                        className="absolute inset-0 rounded-full bg-red-500 opacity-20"
                      />
                    )}
                    <Mic className="w-16 h-16 text-white" />
                  </motion.button>
                )}
              </div>

              <p className="text-center text-gray-600 mt-4">
                {recording
                  ? "Recording... Tap to stop"
                  : evaluating
                  ? "Processing your pronunciation..."
                  : "Tap to record"}
              </p>
            </div>
          </motion.div>
        )}

        {/* ---------------------------------------------------------
            STEP 4: RESULTS
        ---------------------------------------------------------- */}
        {step === "results" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center"
          >
            <h2 className="text-3xl font-bold mb-8">Battle Results!</h2>

            <div className="grid md:grid-cols-2 gap-6 mb-8 max-w-3xl mx-auto">
              {/* Player 1 */}
              <motion.div
                initial={{ x: -50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className={`p-8 rounded-2xl shadow-xl ${
                  player1Score > player2Score
                    ? "bg-gradient-to-br from-yellow-400 to-orange-500 text-white ring-4 ring-yellow-300"
                    : "bg-white"
                }`}
              >
                <h3 className="text-2xl font-bold mb-3">{player1Name}</h3>
                <p className="text-6xl font-bold">{player1Score}%</p>
                {player1Score > player2Score && (
                  <p className="mt-2 text-lg font-semibold">🏆 Winner!</p>
                )}
              </motion.div>

              {/* Player 2 */}
              <motion.div
                initial={{ x: 50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.4 }}
                className={`p-8 rounded-2xl shadow-xl ${
                  player2Score > player1Score
                    ? "bg-gradient-to-br from-yellow-400 to-orange-500 text-white ring-4 ring-yellow-300"
                    : "bg-white"
                }`}
              >
                <h3 className="text-2xl font-bold mb-3">{player2Name}</h3>
                <p className="text-6xl font-bold">{player2Score}%</p>
                {player2Score > player1Score && (
                  <p className="mt-2 text-lg font-semibold">🏆 Winner!</p>
                )}
              </motion.div>
            </div>

            {/* Tie Message */}
            {player1Score === player2Score && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.6 }}
                className="mb-8 p-6 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-2xl inline-block shadow-xl"
              >
                <p className="text-2xl font-bold">🤝 It's a Tie!</p>
              </motion.div>
            )}

            {/* Winner Announcement */}
            {player1Score !== player2Score && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.6, type: "spring" }}
                className="mb-8"
              >
                <div className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-yellow-400 via-orange-500 to-pink-500 text-white rounded-2xl shadow-2xl">
                  <Trophy className="w-10 h-10" />
                  <span className="text-3xl font-bold">
                    {player1Score > player2Score ? player1Name : player2Name}{" "}
                    Wins! 🎉
                  </span>
                </div>
              </motion.div>
            )}

            {/* Play Again Button */}
            <button
              onClick={resetBattle}
              className="px-8 py-4 bg-gray-800 text-white text-lg font-bold rounded-xl hover:bg-black transition flex items-center gap-2 mx-auto"
            >
              <RotateCcw className="w-5 h-5" />
              Play Again
            </button>
          </motion.div>
        )}
      </div>
    </div>
  );
}