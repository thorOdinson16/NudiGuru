import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Mic, Volume2, Sparkles } from "lucide-react";

export default function Splash() {
  const navigate = useNavigate();
  const [waveform, setWaveform] = useState([0.3, 0.5, 0.8, 0.5, 0.3, 0.6, 0.9, 0.4, 0.7]);

  // Local replacement for createPageUrl
  const createPageUrl = (p) => "/" + p;

  useEffect(() => {
    const interval = setInterval(() => {
      setWaveform((prev) => prev.map(() => Math.random() * 0.7 + 0.3));
    }, 200);

    const timeout = setTimeout(() => {
      navigate(createPageUrl("Dashboard"));
    }, 3000);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, [navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-yellow-400 via-orange-500 to-pink-500 flex items-center justify-center overflow-hidden relative">
      <motion.div
        animate={{ scale: [1, 1.2, 1], rotate: [0, 180, 360] }}
        transition={{ duration: 20, repeat: Infinity }}
        className="absolute top-20 left-20 w-64 h-64 rounded-full bg-white/10 blur-3xl"
      />
      <motion.div
        animate={{ scale: [1.2, 1, 1.2], rotate: [360, 180, 0] }}
        transition={{ duration: 15, repeat: Infinity }}
        className="absolute bottom-20 right-20 w-96 h-96 rounded-full bg-blue-500/10 blur-3xl"
      />
      <div className="text-center z-10 px-6">
        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: "spring", duration: 1 }}
          className="mb-8 flex justify-center"
        >
          <div className="relative">
            <div className="w-32 h-32 rounded-3xl bg-white shadow-2xl flex items-center justify-center">
              <Mic className="w-16 h-16 text-orange-500" />
            </div>
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="absolute -top-2 -right-2"
            >
              <Sparkles className="w-8 h-8 text-yellow-300" />
            </motion.div>
          </div>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="text-6xl md:text-7xl font-bold text-white mb-4 tracking-tight kannada-text"
        >
          NudiGuru
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="text-xl md:text-2xl text-white/90 mb-8 font-medium"
        >
          Your AI Kannada Speaking Coach
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="flex items-end justify-center gap-2 h-20"
        >
          {waveform.map((height, i) => (
            <motion.div
              key={i}
              animate={{ scaleY: height }}
              transition={{ duration: 0.2 }}
              className="w-2 bg-white rounded-full origin-bottom"
            />
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="mt-12 flex items-center justify-center gap-2 text-white/80"
        >
          <Volume2 className="w-5 h-5 animate-pulse" />
          <span className="text-sm font-medium">Loading your experience...</span>
        </motion.div>
      </div>
    </div>
  );
}
