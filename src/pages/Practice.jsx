// frontend/src/pages/Practice.jsx
import React, { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Play,
  RotateCcw,
  Volume2,
  Sparkles,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import WaveformVisualizer from "@/components/ui/WaveformVisualizer";

const API_URL = "http://localhost:8000";

// ===========================
// API Functions - FIXED
// ===========================

async function fetchLessons() {
  const response = await fetch(`${API_URL}/lessons`);
  if (!response.ok) throw new Error("Failed to fetch lessons");
  return response.json();
}

async function evaluatePronunciation(audioBlob, lessonId) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");
  formData.append("lesson_id", lessonId); // ✅ FIXED: Send lesson_id
  
  console.log(`📤 Sending evaluation for lesson: ${lessonId}`);
  
  const response = await fetch(`${API_URL}/evaluate`, {
    method: "POST",
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error);
  }
  
  return response.json();
}

// ===========================
// Helper: Convert AudioBuffer to WAV
// ===========================
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

// ===========================
// Main Component
// ===========================
export default function Practice() {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [selectedLesson, setSelectedLesson] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);

  // Fetch Lessons
  const { data: lessons = [], isLoading: lessonsLoading } = useQuery({
    queryKey: ["lessons"],
    queryFn: fetchLessons,
  });

  // Select default lesson
  useEffect(() => {
    if (lessons.length > 0 && !selectedLesson) {
      setSelectedLesson(lessons[0]);
      console.log(`📖 Selected lesson: ${lessons[0].id}`);
    }
  }, [lessons, selectedLesson]);

  // -----------------------------
  // Recording Logic
  // -----------------------------
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();

      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: "audio/webm" });
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });

        try {
          const arrayBuffer = await audioBlob.arrayBuffer();
          const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
          const wavBlob = audioBufferToWav(audioBuffer);
          const audioUrl = URL.createObjectURL(wavBlob);

          setRecordedAudio(audioUrl);
          await analyzeRecording(wavBlob);
        } catch (err) {
          console.error("Audio processing failed:", err);
          setFeedback({
            accuracy_score: 0,
            syllables: [],
            areas_to_improve: ["Audio processing failed. Please try again."],
          });
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      console.log("🎤 Recording started");
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Please allow microphone access.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
      console.log("🛑 Recording stopped");
    }
  };

  // -----------------------------
  // Evaluation
  // -----------------------------
  const analyzeRecording = async (wavBlob) => {
    if (!selectedLesson) {
      console.error("❌ No lesson selected!");
      return;
    }
    
    setIsEvaluating(true);
    
    try {
      console.log(`🔍 Analyzing for lesson: ${selectedLesson.id}`);
      const feedbackData = await evaluatePronunciation(wavBlob, selectedLesson.id);
      console.log("✅ Evaluation complete:", feedbackData);
      setFeedback(feedbackData);
    } catch (error) {
      console.error("❌ Evaluation failed:", error);
      setFeedback({
        accuracy_score: 0,
        syllables: [],
        areas_to_improve: ["Could not process audio. Please try again."],
      });
    } finally {
      setIsEvaluating(false);
    }
  };

  // -----------------------------
  // Play Native Audio - FIXED
  // -----------------------------
  const playNativeAudio = async () => {
    if (!selectedLesson) return;
    
    try {
      setIsPlaying(true);
      console.log(`🔊 Playing audio for: ${selectedLesson.id}`);
      
      const response = await fetch(`${API_URL}/tts/generate/${selectedLesson.id}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch audio');
      }
      
      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      
      const audio = new Audio(audioUrl);
      
      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
        console.log("✅ Audio playback finished");
      };
      
      audio.onerror = (e) => {
        setIsPlaying(false);
        console.error("❌ Audio playback error:", e);
        alert('Failed to play audio');
      };
      
      await audio.play();
    } catch (err) {
      console.error('❌ TTS playback error:', err);
      setIsPlaying(false);
      alert('Could not play reference audio');
    }
  };

  const reset = () => {
    setRecordedAudio(null);
    setFeedback(null);
    console.log("🔄 Reset state");
  };

  // -----------------------------
  // Loading State
  // -----------------------------
  if (lessonsLoading || !selectedLesson) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500 mx-auto mb-4" />
          <p className="text-gray-600">Loading lessons...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Title */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Practice Pronunciation
          </h1>
          <p className="text-gray-600">
            Speak clearly and get instant AI feedback
          </p>
        </motion.div>

        {/* Lesson Selector - FIXED */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <Card className="glass-card border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex gap-2 overflow-x-auto pb-2">
                {lessons.map((lesson) => (
                  <Button
                    key={lesson.id}
                    variant={selectedLesson?.id === lesson.id ? "default" : "outline"}
                    onClick={() => {
                      console.log(`🎯 Switching to lesson: ${lesson.id}`);
                      setSelectedLesson(lesson);
                      reset();
                    }}
                    className={`whitespace-nowrap transition-all ${
                      selectedLesson?.id === lesson.id
                        ? "bg-gradient-to-r from-orange-500 to-pink-500 text-white"
                        : ""
                    }`}
                  >
                    Lesson {lesson.order}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Practice Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mb-6"
        >
          <Card className="glass-card border-0 shadow-2xl">
            <CardHeader className="text-center pb-4">
              <CardTitle className="text-sm text-gray-600 mb-4">
                {selectedLesson.title} (Lesson {selectedLesson.order})
              </CardTitle>

              <div className="space-y-4">
                <p className="text-5xl font-bold kannada-text text-gray-900">
                  {selectedLesson.kannada_text}
                </p>
                <p className="text-xl text-gray-700 italic">
                  {selectedLesson.transliteration}
                </p>
                <p className="text-base text-gray-600">
                  "{selectedLesson.english_translation}"
                </p>
              </div>
            </CardHeader>

            <CardContent className="space-y-6">
              {/* Native Audio */}
              <div className="flex justify-center">
                <Button
                  onClick={playNativeAudio}
                  variant="outline"
                  className="flex items-center gap-2"
                  disabled={isRecording || isPlaying}
                >
                  <Volume2 className="w-5 h-5" />
                  {isPlaying ? "Playing..." : "Play Native Audio"}
                </Button>
              </div>

              {/* Waveform */}
              <div className="bg-gradient-to-r from-orange-50 to-pink-50 rounded-2xl p-4">
                <WaveformVisualizer
                  isRecording={isRecording}
                  isPlaying={isPlaying}
                  height={100}
                />
              </div>

              {/* Record Button */}
              <div className="flex justify-center">
                <motion.button
                  whileTap={{ scale: 0.95 }}
                  whileHover={{ scale: 1.05 }}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={!!feedback || isEvaluating}
                  className={`relative w-32 h-32 rounded-full flex items-center justify-center shadow-2xl transition-all ${
                    isRecording
                      ? "bg-gradient-to-br from-red-500 to-pink-600"
                      : feedback || isEvaluating
                      ? "bg-gray-300 cursor-not-allowed"
                      : "bg-gradient-to-br from-orange-500 to-pink-500 hover:shadow-xl"
                  }`}
                >
                  {isRecording && (
                    <motion.div
                      animate={{ scale: [1, 1.3, 1] }}
                      transition={{ repeat: Infinity, duration: 1.5 }}
                      className="absolute inset-0 rounded-full bg-red-500 opacity-20"
                    />
                  )}

                  <Mic className="w-16 h-16 text-white" />
                </motion.button>
              </div>

              <p className="text-center text-sm text-gray-600">
                {isRecording
                  ? "Recording... Tap to stop"
                  : isEvaluating
                  ? "Evaluating your pronunciation..."
                  : feedback
                  ? "Recorded – Review feedback below"
                  : "Tap to record your pronunciation"}
              </p>

              {/* Feedback */}
              <AnimatePresence>
                {feedback && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-4"
                  >
                    {/* Score */}
                    <div className="text-center">
                      <div
                        className={`inline-flex items-center gap-2 px-6 py-3 rounded-2xl text-white shadow-lg ${
                          feedback.accuracy_score >= 85
                            ? "bg-gradient-to-r from-green-400 to-emerald-500"
                            : feedback.accuracy_score >= 60
                            ? "bg-gradient-to-r from-yellow-400 to-orange-500"
                            : "bg-gradient-to-r from-red-400 to-pink-500"
                        }`}
                      >
                        {feedback.accuracy_score >= 85 ? (
                          <CheckCircle2 className="w-6 h-6" />
                        ) : (
                          <AlertCircle className="w-6 h-6" />
                        )}
                        <span className="text-3xl font-bold">
                          {feedback.accuracy_score}%
                        </span>
                      </div>
                      <p className="text-gray-600 mt-2">Accuracy Score</p>
                    </div>

                    {/* Syllables */}
                    {feedback.syllables && feedback.syllables.length > 0 && (
                      <div>
                        <p className="text-sm font-medium text-gray-700 mb-2">
                          Pronunciation Breakdown:
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {feedback.syllables.map((syllable, idx) => (
                            <div
                              key={idx}
                              className="px-4 py-2 rounded-lg text-sm font-medium"
                              style={{
                                backgroundColor:
                                  syllable.accuracy >= 80
                                    ? "#86EFAC"
                                    : syllable.accuracy >= 60
                                    ? "#FCD34D"
                                    : "#FCA5A5",
                                color: "#1F2937",
                              }}
                            >
                              {syllable.text} ({syllable.accuracy}%)
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Tips */}
                    {feedback.areas_to_improve && feedback.areas_to_improve.length > 0 && (
                      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
                        <p className="font-medium text-yellow-900 mb-1">
                          Tips to Improve:
                        </p>
                        <ul className="text-sm text-yellow-800 list-disc list-inside space-y-1">
                          {feedback.areas_to_improve.map((tip, i) => (
                            <li key={i}>{tip}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Play Reference */}
                    <div className="flex justify-center gap-4">
                      <Button
                        onClick={playNativeAudio}
                        variant="outline"
                        className="border-blue-300 text-blue-700 flex items-center gap-2"
                      >
                        <Volume2 className="w-4 h-4" />
                        Hear Correct Pronunciation
                      </Button>

                      <Button
                        onClick={reset}
                        className="bg-gradient-to-r from-orange-500 to-pink-500 text-white hover:shadow-lg"
                      >
                        <RotateCcw className="w-4 h-4 mr-2" />
                        Try Again
                      </Button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}