import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

export default function WaveformVisualizer({
  isRecording = false,
  isPlaying = false,
  height = 80,
  barsCount = 40,
}) {
  const [bars, setBars] = useState(
    Array.from({ length: barsCount }, () => 0.2)
  );

  useEffect(() => {
    // If neither recording nor playing → reset bars
    if (!isRecording && !isPlaying) {
      setBars(Array.from({ length: barsCount }, () => 0.2));
      return;
    }

    // Active waveform animation
    const interval = setInterval(() => {
      setBars(
        Array.from({ length: barsCount }, () => Math.random() * 0.8 + 0.2)
      );
    }, 100);

    return () => clearInterval(interval);
  }, [isRecording, isPlaying, barsCount]);

  return (
    <div
      className="flex items-center justify-center gap-[3px] px-4"
      style={{ height }}
    >
      {bars.map((scale, index) => (
        <motion.div
          key={index}
          animate={{ scaleY: isRecording || isPlaying ? scale : 0.2 }}
          transition={{ duration: 0.12, ease: "easeOut" }}
          className={`
            w-1 rounded-full origin-bottom 
            ${
              isRecording
                ? "bg-gradient-to-t from-red-500 to-pink-500"
                : isPlaying
                ? "bg-gradient-to-t from-blue-500 to-cyan-500"
                : "bg-gray-300"
            }
          `}
          style={{
            height: "100%",
          }}
        />
      ))}
    </div>
  );
}
