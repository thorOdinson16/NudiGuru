import React from "react";
import { motion } from "framer-motion";
import {
  TrendingUp,
  BookOpen,
  Gauge,
  Flame,
} from "lucide-react";

export default function BottomStats({ userStats }) {
  const {
    totalPractices,
    uniqueLessons,
    avgAccuracy,
    streak,
  } = userStats;

  const accuracyColor =
    avgAccuracy >= 90
      ? "text-green-600"
      : avgAccuracy >= 70
      ? "text-yellow-600"
      : "text-red-600";

  const containerVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
  };

  const item = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="fixed bottom-0 left-0 right-0 px-6 pb-6 z-40"
    >
      <motion.div
        variants={item}
        className="
          max-w-5xl mx-auto
          bg-white/70 backdrop-blur-xl 
          border border-white/40
          rounded-3xl shadow-lg 
          p-6 md:p-8
        "
      >
        <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
          <TrendingUp className="text-orange-500" />
          ನಿಮ್ಮ ಪ್ರಯಾಣ / Your Journey
        </h2>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {/* Total Practices */}
          <motion.div variants={item} className="text-center">
            <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-orange-100 flex items-center justify-center">
              <TrendingUp className="text-orange-500" />
            </div>
            <div className="text-3xl font-bold">{totalPractices}</div>
            <p className="text-sm text-gray-600">Total Practices</p>

            {/* Tiny Progress Highlight */}
            <div className="w-full h-2 mt-2 bg-gray-200 rounded-full">
              <motion.div
                className="h-2 bg-orange-500 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(totalPractices, 100)}%` }}
                transition={{ duration: 1 }}
              />
            </div>
          </motion.div>

          {/* Unique Lessons */}
          <motion.div variants={item} className="text-center">
            <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-green-100 flex items-center justify-center">
              <BookOpen className="text-green-600" />
            </div>
            <div className="text-3xl font-bold">{uniqueLessons}</div>
            <p className="text-sm text-gray-600">Unique Lessons</p>
          </motion.div>

          {/* Average Accuracy */}
          <motion.div variants={item} className="text-center">
            <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-blue-100 flex items-center justify-center">
              <Gauge className="text-blue-600" />
            </div>
            <div className={`text-3xl font-bold ${accuracyColor}`}>
              {avgAccuracy}%
            </div>
            <p className="text-sm text-gray-600">Avg Accuracy</p>
          </motion.div>

          {/* Streak */}
          <motion.div variants={item} className="text-center">
            <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-red-100 flex items-center justify-center">
              <Flame className="text-red-500" />
            </div>
            <div className="text-3xl font-bold">{streak}</div>
            <p className="text-sm text-gray-600">Day Streak</p>
          </motion.div>
        </div>
      </motion.div>
    </motion.div>
  );
}
