// frontend/src/pages/Practice.jsx
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  RotateCcw,
  Volume2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import WaveformVisualizer from "@/components/ui/WaveformVisualizer";
import { fetchLessons, evaluatePronunciation, getTtsUrl } from "@/api/client";
import { audioBufferToWav } from "@/lib/audio";

export default function Practice() {
  const [searchParams] = useSearchParams();
  const requestedLesson = searchParams.get("lesson");

  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [selectedLesson, setSelectedLesson] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const audioRef = useRef(null);

  const { data: lessons = [], isLoading: lessonsLoading } = useQuery({
    queryKey: ["lessons"],
    queryFn: fetchLessons,
  });

  useEffect(() => {
    if (lessons.length === 0) return;
    setSelectedLesson((current) => {
      if (current) return current;
      const match = lessons.find((l) => l.id === requestedLesson);
      return match || lessons[0];
    });
  }, [lessons, requestedLesson]);

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
          await analyzeRecording(wavBlob);
        } catch {
          setFeedback({
            accuracy_score: 0,
            syllables: [],
            areas_to_improve: ["Audio processing failed. Please try again."],
          });
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch {
      alert("Please allow microphone access.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
    }
  };

  const analyzeRecording = async (wavBlob) => {
    if (!selectedLesson) return;
    setIsEvaluating(true);
    try {
      const feedbackData = await evaluatePronunciation(wavBlob, selectedLesson.id);
      setFeedback(feedbackData);
    } catch {
      setFeedback({
        accuracy_score: 0,
        syllables: [],
        areas_to_improve: ["Could not process audio. Please try again."],
      });
    } finally {
      setIsEvaluating(false);
    }
  };

  const playNativeAudio = () => {
    if (!selectedLesson) return;
    setIsPlaying(true);

    if (audioRef.current) {
      audioRef.current.pause();
    }
    const audio = new Audio(getTtsUrl(selectedLesson.id));
    audioRef.current = audio;

    audio.onended = () => setIsPlaying(false);
    audio.onerror = () => {
      setIsPlaying(false);
      alert("Could not play reference audio");
    };

    audio.play().catch(() => {
      setIsPlaying(false);
      alert("Could not play reference audio");
    });
  };

  const reset = () => {
    setFeedback(null);
  };

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
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Practice Pronunciation</h1>
          <p className="text-gray-600">Speak clearly and get instant AI feedback</p>
        </motion.div>

        {/* Lesson Selector */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          <Card className="glass-card border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex gap-2 overflow-x-auto pb-2">
                {lessons.map((lesson) => (
                  <Button
                    key={lesson.id}
                    variant={selectedLesson?.id === lesson.id ? "default" : "outline"}
                    onClick={() => {
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
        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="mb-6">
          <Card className="glass-card border-0 shadow-2xl">
            <CardHeader className="text-center pb-4">
              <CardTitle className="text-sm text-gray-600 mb-4">
                {selectedLesson.title} (Lesson {selectedLesson.order})
              </CardTitle>
              <div className="space-y-4">
                <p className="text-5xl font-bold kannada-text text-gray-900">
                  {selectedLesson.kannada_text}
                </p>
                <p className="text-xl text-gray-700 italic">{selectedLesson.transliteration}</p>
                <p className="text-base text-gray-600">"{selectedLesson.english_translation}"</p>
              </div>
            </CardHeader>

            <CardContent className="space-y-6">
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

              <div className="bg-gradient-to-r from-orange-50 to-pink-50 rounded-2xl p-4">
                <WaveformVisualizer isRecording={isRecording} isPlaying={isPlaying} height={100} />
              </div>

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

              <AnimatePresence>
                {feedback && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-4"
                  >
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
                        <span className="text-3xl font-bold">{feedback.accuracy_score}%</span>
                      </div>
                      <p className="text-gray-600 mt-2">Accuracy Score</p>
                    </div>

                    {feedback.syllables && feedback.syllables.length > 0 && (
                      <div>
                        <p className="text-sm font-medium text-gray-700 mb-2">Pronunciation Breakdown:</p>
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

                    {feedback.areas_to_improve && feedback.areas_to_improve.length > 0 && (
                      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
                        <p className="font-medium text-yellow-900 mb-1">Tips to Improve:</p>
                        <ul className="text-sm text-yellow-800 list-disc list-inside space-y-1">
                          {feedback.areas_to_improve.map((tip, i) => (
                            <li key={i}>{tip}</li>
                          ))}
                        </ul>
                      </div>
                    )}

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
