# frontend/src/components/ui/badge.jsx
export function Badge({children}){return <span>{children}</span>}

# frontend/src/components/ui/BottomStats.jsx
import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  TrendingUp,
  BookOpen,
  Gauge,
  Flame,
} from "lucide-react";

export default function BottomStats({ userStats }) {
  const [isVisible, setIsVisible] = useState(false);

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

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.innerHeight + window.scrollY;
      const threshold = document.body.offsetHeight - 100; // 200px from bottom
      setIsVisible(scrollPosition >= threshold);
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  if (!isVisible) return null;

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

# frontend/src/components/ui/button.jsx
export function Button({children,...p}){return <button {...p}>{children}</button>}

# frontend/src/components/ui/card.jsx
import * as React from "react";
import { cn } from "@/lib/utils";

const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-xl border bg-card text-card-foreground shadow",
      className
    )}
    {...props}
  />
));
Card.displayName = "Card";

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn("text-2xl font-semibold leading-none tracking-tight", className)}
    {...props}
  />
));
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
));
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
));
CardFooter.displayName = "CardFooter";

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
};

# frontend/src/components/ui/Header.jsx
import React from "react";
import { NavLink } from "react-router-dom";
import { Home, Mic, Library, User } from "lucide-react";


export default function Header() {
  return (
    <header className="w-full bg-white/80 backdrop-blur-md shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">

        {/* Logo Left */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-yellow-400 to-orange-500 text-white flex items-center justify-center text-2xl font-bold">
            ನು
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">NudiGuru</h1>
            <p className="text-sm text-gray-500">AI Kannada Speaking Coach</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-6">
          <NavLink
            to="/Dashboard"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition 
              ${isActive ? "bg-gradient-to-r from-yellow-400 to-orange-500 text-white shadow-md" : "text-gray-700"}`
            }
          >
            <Home size={18} /> Dashboard
          </NavLink>

          <NavLink
            to="/Practice"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition 
              ${isActive ? "bg-gradient-to-r from-orange-500 to-pink-500 text-white shadow-md" : "text-gray-700"}`
            }
          >
            <Mic size={18} /> Practice
          </NavLink>

          <NavLink
            to="/Lessons"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition 
              ${isActive ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-md" : "text-gray-700"}`
            }
          >
            <Library size={18} /> Lessons
          </NavLink>

          
        </nav>
      </div>
    </header>
  );
}

# frontend/src/components/ui/progress.jsx
export function Progress({value}){return <div style={{width:value+'%'}}></div>}

# frontend/src/components/ui/ProgressRing.jsx
import React from "react";
import { motion } from "framer-motion";

export default function ProgressRing({ progress, total, size = 120, strokeWidth = 8, color = "#FCD34D" }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const percentage = (progress / total) * 100;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#E5E7EB"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: "easeInOut" }}
        />
      </svg>
      
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.5, type: "spring" }}
          className="text-center"
        >
          <div className="text-3xl font-bold text-gray-900">{progress}</div>
          <div className="text-sm text-gray-600">of {total}</div>
        </motion.div>
      </div>
    </div>
  );

# frontend/src/components/ui/WaveformVisualizer.jsx
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

# frontend/src/lib/utils.js
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

# frontend/src/pages/BottomStats.jsx
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

# frontend/src/pages/Dashboard.jsx
import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { fetchUserStats } from '@/api/client';

import {
  Mic,
  User,
  Library,
  Flame,
  Target,
  Award,
  ChevronRight,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ProgressRing from "@/components/ui/ProgressRing";
import BottomStats from "@/components/ui/BottomStats";

// Create page URL helper
const createPageUrl = (page) => "/" + page;

/* ---------------------------------------------------------
   TEMP DATA (Replace with FastAPI later)
---------------------------------------------------------- */

// Replace fetchUser:
const { data: user } = useQuery({
  queryKey: ["currentUser"],
  queryFn: fetchUserStats,
});

const fetchPracticeSessions = async () => {
  return [
    { accuracy_score: 95 },
    { accuracy_score: 88 },
    { accuracy_score: 92 },
  ];
};

const fetchVoiceTwin = async () => {
  return { status: "ready" };
};

/* --------------------------------------------------------- */

export default function Dashboard() {
  const [greeting, setGreeting] = useState("");

  // Queries
  const { data: user } = useQuery({
    queryKey: ["currentUser"],
    queryFn: fetchUser,
  });

  const { data: sessions = [] } = useQuery({
    queryKey: ["practiceSessions"],
    queryFn: fetchPracticeSessions,
  });

  const { data: voiceTwin } = useQuery({
    queryKey: ["voiceTwin"],
    queryFn: fetchVoiceTwin,
  });

  // Greeting
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting("ಶುಭೋದಯ");
    else if (hour < 17) setGreeting("ಶುಭ ಮಧ್ಯಾಹ್ನ");
    else setGreeting("ಶುಭ ಸಂಜೆ");
  }, []);

  const dailyProgress = user?.daily_practice_count || 0;
  const dailyGoal = user?.daily_goal || 10;
  const streak = user?.streak_days || 0;
  const perfectSessions = sessions.filter((s) => s.accuracy_score >= 90).length;

  // Animation
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <>
      <div className="min-h-screen pb-40">
        <div className="max-w-6xl mx-auto px-4 py-8">

          {/* Greeting */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-2 kannada-text">
              {greeting}, {user?.full_name || "Friend"}!
            </h1>

            <p className="text-lg text-gray-600">
              Ready to practice your Kannada pronunciation today?
            </p>
          </motion.div>

          {/* Stats */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid md:grid-cols-3 gap-6 mb-12"
          >
            {/* Progress */}
            <motion.div variants={itemVariants}>
              <Card className="glass-card border-0 shadow-lg hover:shadow-xl">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5 text-orange-500" />
                    ಇಂದಿನ ಪ್ರಗತಿ / Today’s Progress
                  </CardTitle>
                </CardHeader>

                <CardContent className="flex flex-col items-center pt-2">
                  <ProgressRing
                    progress={dailyProgress}
                    total={dailyGoal}
                    size={140}
                    color="url(#gradient1)"
                  />

                  <svg width="0" height="0">
                    <defs>
                      <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#FCD34D" />
                        <stop offset="100%" stopColor="#F97316" />
                      </linearGradient>
                    </defs>
                  </svg>

                  <p className="text-sm text-gray-600 mt-4">practices today</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* Streak */}
            <motion.div variants={itemVariants}>
              <Card className="glass-card border-0 shadow-lg hover:shadow-xl h-full">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2">
                    <Flame className="w-5 h-5 text-orange-500" />
                    ಪ್ರಸ್ತುತ / Current Streak
                  </CardTitle>
                </CardHeader>

                <CardContent className="flex flex-col items-center justify-center flex-1">
                  <div className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-500 to-pink-500">
                    {streak}
                  </div>
                  <p className="text-sm text-gray-600 mt-2">days in a row</p>
                </CardContent>
              </Card>
            </motion.div>

            {/* Perfect Scores */}
            <motion.div variants={itemVariants}>
              <Card className="glass-card border-0 shadow-lg hover:shadow-xl h-full">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2">
                    <Award className="w-5 h-5 text-blue-500" />
                    ಗಳಿಸಿದ ಅಂಕಗಳು / Scores Gained
                  </CardTitle>
                </CardHeader>

                <CardContent className="flex flex-col items-center justify-center flex-1">
                  <div className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-cyan-500">
                    {perfectSessions}
                  </div>
                  <p className="text-sm text-gray-600 mt-2">lessons mastered</p>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>

          {/* ------------------------------- */}
          {/*   J O U R N E Y   S E C T I O N */}
          {/* ------------------------------- */}

          <div id="journey-section" className="scroll-mt-32 mb-6">
            <motion.h2
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-3xl font-bold text-gray-900 mb-4"
            >
              ನಿಮ್ಮ ಕಲಿಕೆಯ ಪ್ರಯಾಣ / Your Learning Journey
            </motion.h2>

            <p className="text-gray-600 mb-8">
              Continue where you left off — choose your next learning action.
            </p>
          </div>

          {/* Action Cards */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid md:grid-cols-3 gap-6"
          >
            {/* Practice */}
            <motion.div variants={itemVariants}>
              <Link to={createPageUrl("Practice")}>
                <Card className="glass-card border-0 shadow-lg hover:shadow-2xl transition-all hover:scale-105 cursor-pointer group h-full relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-orange-500/10 to-pink-500/10 opacity-0 group-hover:opacity-100 transition-opacity" />

                  <CardContent className="p-8 flex flex-col items-center text-center relative z-10">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-orange-400 to-pink-500 flex items-center justify-center mb-4 shadow-lg">
                      <Mic className="w-10 h-10 text-white" />
                    </div>

                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      ಈಗ ಕಲಿಯೋಣ / Practice Now
                    </h3>

                    <p className="text-gray-600 mb-4">
                      Start practicing your pronunciation with instant feedback
                    </p>

                    <ChevronRight className="w-6 h-6 text-gray-400 group-hover:text-orange-500" />
                  </CardContent>
                </Card>
              </Link>
            </motion.div>

            

            {/* Lessons */}
            <motion.div variants={itemVariants}>
              <Link to={createPageUrl("Lessons")}>
                <Card className="glass-card border-0 shadow-lg hover:shadow-2xl transition-all hover:scale-105 cursor-pointer group h-full relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 to-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity" />

                  <CardContent className="p-8 flex flex-col items-center text-center relative z-10">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center mb-4 shadow-lg">
                      <Library className="w-10 h-10 text-white" />
                    </div>

                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      ಪಾಠಗಳು / Lessons
                    </h3>

                    <p className="text-gray-600 mb-4">
                      Browse 15 benchmark lessons at your own pace
                    </p>

                    <ChevronRight className="w-6 h-6 text-gray-400 group-hover:text-green-500" />
                  </CardContent>
                </Card>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* BOTTOM FIXED STATS */}
      <BottomStats
        userStats={{
          totalPractices: user?.total_practices || 0,
          uniqueLessons: sessions?.length || 0,
          avgAccuracy: Math.round(
            sessions.length
              ? sessions.reduce((s, x) => s + x.accuracy_score, 0) /
                  sessions.length
              : 0
          ),
          streak: user?.streak_days || 0,
        }}
      />
    </>
  );
}

# frontend/src/pages/Lessons.jsx
import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { CheckCircle2, Circle, Target, Play, Lock } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// TEMP createPageUrl (remove when FastAPI routing ready)
const createPageUrl = (p) => "/" + p;

// TEMP API placeholders (replace with FastAPI fetch)
const fetchLessons = async () => {
  // Replace with FastAPI GET: /lessons
  return [
    {
      id: 1,
      order: 1,
      title: "Greeting",
      kannada_text: "ನಮಸ್ಕಾರ",
      transliteration: "Namaskāra",
      difficulty: "beginner",
    },
    {
      id: 2,
      order: 2,
      title: "How are you?",
      kannada_text: "ಹೇಗಿದ್ದೀಯ?",
      transliteration: "Hēgiddīya?",
      difficulty: "beginner",
    },
  ];
};

const fetchPracticeSessions = async () => {
  // Replace with FastAPI GET: /practice-sessions
  return [];
};

export default function Lessons() {
  const { data: lessons = [], isLoading } = useQuery({
    queryKey: ["lessons"],
    queryFn: fetchLessons,
  });

  const { data: sessions = [] } = useQuery({
    queryKey: ["practiceSessions"],
    queryFn: fetchPracticeSessions,
  });

  const getLessonStatus = (lessonId) => {
    const lessonSessions = sessions.filter((s) => s.lesson_id === lessonId);
    if (lessonSessions.length === 0) return "not_attempted";

    const bestScore = Math.max(...lessonSessions.map((s) => s.accuracy_score || 0));
    if (bestScore >= 90) return "perfected";
    if (bestScore >= 75) return "practiced";
    return "attempted";
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "perfected":
        return <CheckCircle2 className="w-6 h-6 text-green-500" />;
      case "practiced":
        return <Target className="w-6 h-6 text-yellow-500" />;
      case "attempted":
        return <Circle className="w-6 h-6 text-gray-400" />;
      default:
        return <Circle className="w-6 h-6 text-gray-300" />;
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "perfected":
        return <Badge className="bg-green-100 text-green-800">Perfected</Badge>;
      case "practiced":
        return <Badge className="bg-yellow-100 text-yellow-800">Practiced</Badge>;
      case "attempted":
        return <Badge className="bg-blue-100 text-blue-800">Attempted</Badge>;
      default:
        return <Badge variant="outline">Not Started</Badge>;
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case "beginner":
        return "bg-green-100 text-green-800";
      case "intermediate":
        return "bg-yellow-100 text-yellow-800";
      case "advanced":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500 mx-auto mb-4" />
          <p className="text-gray-600">Loading lessons...</p>
        </div>
      </div>
    );
  }

  const stats = {
    total: lessons.length,
    perfected: lessons.filter((l) => getLessonStatus(l.id) === "perfected").length,
    practiced: lessons.filter((l) => getLessonStatus(l.id) === "practiced").length,
    attempted: lessons.filter((l) => getLessonStatus(l.id) === "attempted").length,
  };

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-6xl mx-auto px-4">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Benchmark Lessons</h1>
          <p className="text-lg text-gray-600">Master essential Kannada phrases</p>
        </motion.div>

        {/* Stats */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <Card><CardContent className="p-6 text-center"><div className="text-3xl font-bold">{stats.total}</div><div>Total Lessons</div></CardContent></Card>
          <Card><CardContent className="p-6 text-center"><div className="text-3xl font-bold text-green-600">{stats.perfected}</div><div>Perfected</div></CardContent></Card>
          <Card><CardContent className="p-6 text-center"><div className="text-3xl font-bold text-yellow-600">{stats.practiced}</div><div>Practiced</div></CardContent></Card>
          <Card><CardContent className="p-6 text-center"><div className="text-3xl font-bold text-blue-600">{stats.attempted}</div><div>Attempted</div></CardContent></Card>
        </motion.div>

        {/* List */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="space-y-4">
          {lessons.map((lesson, index) => {
            const status = getLessonStatus(lesson.id);
            return (
              <motion.div key={lesson.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.05 }}>
                <Link to={createPageUrl("Practice")}>
                  <Card className="shadow-lg hover:scale-105 transition cursor-pointer">
                    <CardContent className="p-6 flex gap-4">
                      <div>{getStatusIcon(status)}</div>
                      <div className="w-12 h-12 bg-orange-400 text-white rounded-xl flex items-center justify-center">{lesson.order}</div>
                      <div className="flex-1">
                        <div className="flex gap-2">
                          <h3 className="font-semibold">{lesson.title}</h3>
                          {getStatusBadge(status)}
                          <Badge className={getDifficultyColor(lesson.difficulty)}>{lesson.difficulty}</Badge>
                        </div>
                        <p className="text-2xl kannada-text">{lesson.kannada_text}</p>
                        <p className="text-sm italic">{lesson.transliteration}</p>
                      </div>
                      <div><Play /></div>
                    </CardContent>
                  </Card>
                </Link>
              </motion.div>
            );
          })}
        </motion.div>

        {lessons.length === 0 && (
          <div className="text-center py-12">
            <Lock className="w-16 h-16 mx-auto text-gray-400" />
            <h3>No Lessons Yet</h3>
          </div>
        )}
      </div>
    </div>
  );
}

# frontend/src/pages/Practice.jsx
import React, { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
import { Progress } from "@/components/ui/progress";
import WaveformVisualizer from "@/components/ui/WaveformVisualizer";

// Real API imports
import {
  fetchLessons,
  evaluatePronunciation,
  getNativeAudioUrl,
} from "@/api/client";

// Local createPageUrl helper
const createPageUrl = (page) => "/" + page;

// ------------------------------
// FASTAPI-READY PLACEHOLDER API (REMOVED - USING REAL API)
// ------------------------------

// ------------------------------
// Helper: Convert AudioBuffer to WAV (16kHz, mono, PCM)
function audioBufferToWav(audioBuffer) {
  const numberOfChannels = 1;
  const sampleRate = 16000;
  const length = audioBuffer.length;

  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);

  // WAV Header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, length * 2, true);

  // Write audio data
  const channelData = audioBuffer.getChannelData(0);
  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// ------------------------------

export default function Practice() {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [selectedLesson, setSelectedLesson] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);

  const queryClient = useQueryClient();

  // ---- Fetch Lessons (Real API) ----
  const { data: lessons = [], isLoading: lessonsLoading } = useQuery({
    queryKey: ["lessons"],
    queryFn: fetchLessons,
  });

  // ---- Select default lesson ----
  useEffect(() => {
    if (lessons.length > 0 && !selectedLesson) {
      setSelectedLesson(lessons[0]);
    }
  }, [lessons, selectedLesson]);

  // -----------------------------
  // Recording Logic
  // -----------------------------
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Initialize AudioContext for conversion later
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });
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
    } catch (err) {
      console.error("Microphone access denied or failed:", err);
      alert("Please allow microphone access to record.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
    }
  };

  // -----------------------------
  // Real Pronunciation Evaluation
  // -----------------------------
  const analyzeRecording = async (wavBlob) => {
    try {
      const feedbackData = await evaluatePronunciation(wavBlob, selectedLesson.id);
      setFeedback(feedbackData);
    } catch (error) {
      console.error("Evaluation failed:", error);
      setFeedback({
        accuracy_score: 0,
        syllables: [],
        areas_to_improve: ["Could not process audio. Please try again."],
      });
    }
  };

  // -----------------------------
  // Play Native Audio (Real URL)
  // -----------------------------
  const playNativeAudio = async () => {
    try {
      setIsPlaying(true);
      
      // ✅ Fetch the TTS audio
      const response = await fetch(
        `http://localhost:8000/tts/generate/${selectedLesson.id}`
      );
      
      if (!response.ok) {
        throw new Error('TTS generation failed');
      }
      
      // ✅ Convert to blob and create object URL
      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      
      const audio = new Audio(audioUrl);
      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl); // ✅ Clean up
      };
      
      audio.onerror = () => {
        setIsPlaying(false);
        alert('Failed to play audio');
      };
      
      await audio.play();
    } catch (err) {
      console.error('TTS playback error:', err);
      setIsPlaying(false);
      alert('Could not play reference audio');
    }
  };

  const reset = () => {
    setRecordedAudio(null);
    setFeedback(null);
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

        {/* Lesson Selector */}
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
                    variant={
                      selectedLesson?.id === lesson.id ? "default" : "outline"
                    }
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
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mb-6"
        >
          <Card className="glass-card border-0 shadow-2xl">
            <CardHeader className="text-center pb-4">
              <CardTitle className="text-sm text-gray-600 mb-4">
                {selectedLesson.title}
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
                  Play Native Audio
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
                  disabled={!!feedback}
                  className={`relative w-32 h-32 rounded-full flex items-center justify-center shadow-2xl transition-all ${
                    isRecording
                      ? "bg-gradient-to-br from-red-500 to-pink-600"
                      : feedback
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

                    {/* Syllable Heatmap */}
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
                              {syllable.text}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Improvement Tips */}
                    {feedback.areas_to_improve &&
                      feedback.areas_to_improve.length > 0 && (
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

                    {/* AI Twin (Optional Future Feature) */}
                    <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-4 border border-blue-200">
                      <div className="flex items-start gap-3">
                        <Sparkles className="w-6 h-6 text-blue-500 flex-shrink-0 mt-1" />
                        <div>
                          <p className="font-medium text-gray-900 mb-1">
                            Hear Perfect Pronunciation
                          </p>
                          <p className="text-sm text-gray-600 mb-3">
                            Your AI twin says it perfectly — coming soon!
                          </p>
                          <Button
                            size="sm"
                            variant="outline"
                            className="border-blue-300 text-blue-700 flex items-center gap-2"
                            disabled
                          >
                            <Play className="w-4 h-4" />
                            Play AI Twin (Soon)
                          </Button>
                        </div>
                      </div>
                    </div>
                    
                    {/* 🔊 Hear Correct Pronunciation Button */}
                    <div className="flex justify-center mt-4">
                      <Button
                        onClick={() => playNativeAudio()}
                        variant="outline"
                        className="border-blue-300 text-blue-700 flex items-center gap-2"
                      >
                        <Volume2 className="w-4 h-4" />
                        Hear Correct Pronunciation
                      </Button>
                    </div>

                    {/* Try Again */}
                    <div className="flex justify-center">
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

## src/pages/PronunciationBattlePage.jsx

# frontend/src/pages/Splash.jsx
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

# frontend/src/App.jsx
import { Routes, Route } from "react-router-dom";
import Splash from "./pages/Splash";
import Dashboard from "./pages/Dashboard";
import Practice from "./pages/Practice";
import Lessons from "./pages/Lessons";
import Layout from "./Layout";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Splash />} />
      <Route element={<Layout />}>
        <Route path="/Dashboard" element={<Dashboard />} />
        <Route path="/Practice" element={<Practice />} />
        <Route path="/Lessons" element={<Lessons />} />
      </Route>
    </Routes>
  );
}

# frontend/src/Layout.jsx
import Header from "@/components/ui/Header";
import { Outlet } from "react-router-dom";

export default function Layout() {
  return (
    <>
      <Header />
      <Outlet />
    </>
  );
}

# frontend/src/main.jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./index.css";

const client = new QueryClient();

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>
    </BrowserRouter>
  </React.StrictMode>
);

# frontend/src/queryClient.js
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient();

# backend/HubertPipeline/evaluate_speech.py
#evaluate_speech.py

import os, json
from pydub import AudioSegment
from syllables import WORD_MAP
from features_hubert import extract_embedding
from scorer_hubert import score_syllable

with open("syllable_templates.json") as f:
    templates = json.load(f)

def evaluate(audio_path, word_id):
    syllables = WORD_MAP[word_id]["syllables"]
    audio = AudioSegment.from_wav(audio_path)

    dur = audio.duration_seconds
    syl_dur = dur / len(syllables)

    results = []

    for i, syl in enumerate(syllables):
        start = int(i*syl_dur*1000)
        end   = int((i+1)*syl_dur*1000)

        temp = f"temp_{syl}.wav"
        audio[start:end].export(temp, format="wav")

        emb = extract_embedding(temp)
        sim, ok = score_syllable(word_id, syl, emb, templates)

        results.append({
            "syllable": syl,
            "similarity": sim,
            "correct": ok
        })

        os.remove(temp)

    return results

# backend/HubertPipeline/features_hubert.py
#features_hubert.py

import torch
import numpy as np
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor

#Load PyTorch-only HuBERT components
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model.eval()

def extract_embedding(path):
    audio, sr = librosa.load(path, sr=16000)
    audio, _ = librosa.effects.trim(audio)

    if len(audio) < 2000:
        return np.zeros((768,), dtype=np.float32)

    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state  # shape: (1, T, 768)

    emb = outputs.mean(dim=1).squeeze().numpy()
    emb /= (np.linalg.norm(emb) + 1e-8)

    return emb.astype(np.float32)

# backend/HubertPipeline/preprocess_references.py
#preprocess_references.py

import os, json
from pydub import AudioSegment
from tqdm import tqdm
from syllables import WORD_MAP
from features_hubert import extract_embedding
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

REFERENCE_DIR = "Voices/"
templates = {}

for word_id, info in tqdm(WORD_MAP.items(), desc="Words"):
    syllables = info["syllables"]
    templates[word_id] = {s: [] for s in syllables}

    speakers = [s for s in os.listdir(REFERENCE_DIR)
                if os.path.isdir(os.path.join(REFERENCE_DIR, s))]

    for spk in speakers:
        wav = os.path.join(REFERENCE_DIR, spk, f"{int(word_id[1:])}.wav")
        if not os.path.exists(wav):
            continue

        audio = AudioSegment.from_wav(wav)
        dur = audio.duration_seconds
        syl_dur = dur / len(syllables)

        for i, syl in enumerate(syllables):
            start = int(i * syl_dur * 1000)
            end = int((i+1) * syl_dur * 1000)

            temp = f"temp_{spk}_{word_id}_{syl}.wav"
            audio[start:end].export(temp, format="wav")

            emb = extract_embedding(temp)
            templates[word_id][syl].append(emb.tolist())

            os.remove(temp)

with open("syllable_templates.json", "w") as f:
    json.dump(templates, f, indent=2)

print("DONE → syllable_templates.json")

# backend/HubertPipeline/scorer_hubert.py
#scorer_hubert.py
import numpy as np

def cosine(a, b):
    """Cosine similarity between two vectors"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b + 1e-8))

def score_syllable(word_id, syl, user_emb, templates):
    """
    Score a syllable by comparing user embedding to reference templates
    """
    refs = [np.array(r, dtype=np.float32) for r in templates[word_id][syl]]
    
    if len(refs) == 0:
        return 0.0, False
    
    sims = [cosine(user_emb, r) for r in refs]
    best_sim = max(sims)
    
    threshold = 0.70  # Adjust based on testing
    
    # Convert to percentage-like similarity
    similarity = min(best_sim, 1.0)
    
    return similarity, best_sim >= threshold

# backend/WorkingPipeline/evaluate_speech.py
# evaluate_speech.py
import os
from pydub import AudioSegment
from .syllables import WORD_MAP
from .features_hubert import extract_embedding
from .scorer_hubert import score_syllable
import json

# Load templates
TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "syllable_templates.json"
)

with open(TEMPLATE_PATH) as f:
    templates = json.load(f)

def evaluate(audio_path, word_id):
    syllables = WORD_MAP[word_id]["syllables"]
    audio = AudioSegment.from_wav(audio_path)

    dur = audio.duration_seconds
    syl_dur = dur / len(syllables)

    results = []

    for i, syl in enumerate(syllables):
        start = int(i * syl_dur * 1000)
        end = int((i+1) * syl_dur * 1000)

        temp = f"temp_{syl}.wav"
        audio[start:end].export(temp, format="wav")

        emb = extract_embedding(temp)
        sim, ok = score_syllable(word_id, syl, emb, templates)

        results.append({
            "syllable": syl,
            "similarity": sim,
            "correct": ok
        })

        os.remove(temp)

    return results

# backend/WorkingPipeline/features.py
# features.py

import librosa
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_features(path):
    # Load audio
    y, sr = librosa.load(path, sr=16000)

    # ----------------------------------------
    # 1. Pre-emphasis (boost high frequencies)
    # ----------------------------------------
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # ----------------------------------------
    # 2. Trim silence (important!)
    # ----------------------------------------
    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < 0.1 * sr:  # too short fallback
        return np.zeros((10, 40), dtype=np.float32)

    # ----------------------------------------
    # 3. Log-Mel Spectrogram (40 bins)
    # ----------------------------------------
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=400,        # 25 ms
        hop_length=160,   # 10 ms
        win_length=400,
        n_mels=40,
        fmin=20,
        fmax=7600
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    # ----------------------------------------
    # 4. CMVN Normalization
    # ----------------------------------------
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)

    # ----------------------------------------
    # 5. Return (time, melbins)
    # ----------------------------------------
    return logmel.T.astype(np.float32)

# backend/WorkingPipeline/mel_dtw.py
# mel_dtw.py

import json
import numpy as np
from dtw import dtw
from .features import extract_features
import os

# Path relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "syllable_templates.json")

with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    templates = json.load(f)

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def dtw_dist(a, b):
    return dtw(a, b, dist=lambda x, y: np.linalg.norm(x - y))[0]

def score_syllable(word_id, syl, clip_path):
    # User feature
    user = normalize(extract_features(clip_path))

    # Reference features (all speaker samples)
    refs = [
        normalize(np.array(f, dtype=np.float32))
        for f in templates[word_id][syl]
    ]

    # Compare user to EACH reference
    dists = [dtw_dist(user, r) for r in refs]

    # Best match
    best_dist = min(dists)

    # --------------------------
    # FIXED THRESHOLD
    # --------------------------
    # Good range based on your logs: 500–600
    threshold = 700        # <-- TUNE HERE

    similarity = 1.0 - min(best_dist / threshold, 1.0)

    correct = best_dist < threshold

    return {
        "distance": float(best_dist),
        "similarity": float(similarity),
        "correct": correct
    }

# backend/WorkingPipeline/preprocess_references.py
# preprocess_references.py

import os
import json
from pydub import AudioSegment
from tqdm import tqdm
from syllables import WORD_MAP
from features import extract_features

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REFERENCE_DIR = "Voices/"

templates = {}

# ----------------------------
# Build templates
# ----------------------------
for word_id, info in tqdm(WORD_MAP.items(), desc="Words"):
    syllables = info["syllables"]
    templates[word_id] = {s: [] for s in syllables}

    speakers = [
        s for s in os.listdir(REFERENCE_DIR)
        if os.path.isdir(os.path.join(REFERENCE_DIR, s))
    ]

    for speaker in speakers:
        wav_path = os.path.join(REFERENCE_DIR, speaker, f"{int(word_id[1:])}.wav")
        if not os.path.exists(wav_path):
            continue

        audio = AudioSegment.from_wav(wav_path)

        duration = audio.duration_seconds
        syl_dur = duration / len(syllables)

        # Slice evenly by syllable count
        for i, syl in enumerate(syllables):
            start = int(i * syl_dur * 1000)
            end = int((i + 1) * syl_dur * 1000)

            temp = f"temp_{speaker}_{word_id}_{syl}.wav"
            audio[start:end].export(temp, format="wav")

            feat = extract_features(temp)
            templates[word_id][syl].append(feat.tolist())

            os.remove(temp)

# ----------------------------
# Save templates
# ----------------------------
with open("syllable_templates.json", "w") as f:
    json.dump(templates, f, indent=2)

print("DONE → syllable_templates.json")

# backend/WorkingPipeline/STT.py
#STT.py

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

DURATION = 3
SAMPLING_RATE = 16000

def record_audio(filename="speech.wav"):
    print("ದಯವಿಟ್ಟು ಮಾತಾಡಿ... Recording...")
    audio = sd.rec(int(DURATION * SAMPLING_RATE),
                   samplerate=SAMPLING_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()
    write(filename, SAMPLING_RATE, (audio * 32767).astype(np.int16))
    print("Audio saved:", filename)
    return filename

if __name__ == "__main__":
    audio_file = record_audio()

# backend/WorkingPipeline/syllables.py
WORD_MAP = {
    "w01": {"text": "Namaste", "syllables": ["na", "mas", "te"]},
    "w02": {"text": "Nimma hesarenu", "syllables": ["nim", "ma", "he", "sa", "re", "nu"]},
    "w03": {"text": "Neevu hengiddira", "syllables": ["nee", "vu", "heng", "id", "di", "ra"]},
    "w04": {"text": "Naanu chennagiddini", "syllables": ["naa", "nu", "chen", "na", "gi", "di", "ni"]},
    "w05": {"text": "Adu chennagide", "syllables": ["a", "du", "chen", "na", "gi", "de"]},
    "w06": {"text": "Dhanyavaada", "syllables": ["dha", "nya", "vaa", "da"]},
    "w07": {"text": "Idu eshtu", "syllables": ["i", "du", "es", "htu"]},
    "w08": {"text": "Kimmi maadi", "syllables": ["kim", "mi", "maa", "di"]},
    "w09": {"text": "Dayavittu sahaya maadi", "syllables": ["da", "ya", "vit", "tu", "sa", "ha", "ya", "maa", "di"]},
    "w10": {"text": "Alige hege hogodu", "syllables": ["a", "li", "ge", "he", "ge", "ho", "go", "du"]},
    "w11": {"text": "Naanu Karnataka dalli iddini", "syllables": ["naa", "nu", "kar", "na", "ta", "ka", "dal", "li", "id", "di", "ni"]},
    "w12": {"text": "Naanu kannada kalita iddini", "syllables": ["naa", "nu", "kan", "na", "da", "ka", "li", "ta", "id", "di", "ni"]},
    "w13": {"text": "Oota ayata", "syllables": ["oo", "ta", "ya", "ta"]},
    "w14": {"text": "Nanage gothilla", "syllables": ["na", "na", "ge", "go", "thi", "la"]},
    "w15": {"text": "Manege banni", "syllables": ["ma", "ne", "ge", "ban", "ni"]}
}

# backend/WorkingPipeline/TTS.py
import io
from TTS.utils.synthesizer import Synthesizer
from src.inference import TextToSpeechEngine
from scipy.io.wavfile import write as scipy_wav_write

# ---------------------------
# Load Kannada IndicTTS Model
# ---------------------------

kannada_model = Synthesizer(
    tts_checkpoint="kn/fastpitch/best_model.pth",
    tts_config_path="kn/fastpitch/config.json",
    tts_speakers_file="kn/fastpitch/speakers.pth",   
    tts_languages_file=None,
    vocoder_checkpoint="kn/hifigan/best_model.pth",
    vocoder_config="kn/hifigan/config.json",
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=False
)

# Set up engine
models = {
    "kn": kannada_model
}

engine = TextToSpeechEngine(models)

# ---------------------------
# Generate Kannada TTS
# ---------------------------

DEFAULT_SAMPLING_RATE = 16000

# backend/main.py
# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import shutil
import io
import numpy as np
from scipy.io.wavfile import write as scipy_wav_write, read as scipy_wav_read
import traceback

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

# TTS Setup
TTS_AVAILABLE = False
synthesizer = None

try:
    from TTS.utils.synthesizer import Synthesizer
    
    if os.path.exists("kn/fastpitch/best_model.pth"):
        synthesizer = Synthesizer(
            tts_checkpoint="kn/fastpitch/best_model.pth",
            tts_config_path="kn/fastpitch/config.json",
            tts_speakers_file="kn/fastpitch/speakers.pth",
            vocoder_checkpoint="kn/hifigan/best_model.pth",
            vocoder_config="kn/hifigan/config.json",
            use_cuda=False
        )
        TTS_AVAILABLE = True
        print("✅ TTS Loaded Successfully")
except Exception as e:
    print(f"⚠️ TTS not available: {e}")

app = FastAPI(title="NudiGuru API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    word_id: str = "w01",
    pipeline: str = "both"  # "working", "hubert", or "both"
):
    """
    Evaluate pronunciation using one or both pipelines
    """
    if word_id not in WORD_MAP:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    if not audio.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files accepted")
    
    temp_path = os.path.join(UPLOAD_DIR, f"{word_id}_{audio.filename}")
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        results = {}
        
        # Run Working Pipeline
        if pipeline in ["working", "both"] and WORKING_PIPELINE_AVAILABLE:
            try:
                working_results = evaluate_working(temp_path, word_id)
                similarities = [r["similarity"] for r in working_results]
                working_accuracy = int(sum(similarities) / len(similarities) * 100)
                
                results["working_pipeline"] = {
                    "accuracy": working_accuracy,
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
                results["working_pipeline"] = {"error": str(e)}
        
        # Run HuBERT Pipeline
        if pipeline in ["hubert", "both"] and HUBERT_PIPELINE_AVAILABLE:
            try:
                hubert_results = evaluate_hubert(temp_path, word_id)
                similarities = [r["similarity"] for r in hubert_results]
                hubert_accuracy = int(sum(similarities) / len(similarities) * 100)
                
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
                results["hubert_pipeline"] = {"error": str(e)}
        
        # Combine results if both available
        if "working_pipeline" in results and "hubert_pipeline" in results:
            avg_accuracy = (
                results["working_pipeline"]["accuracy"] + 
                results["hubert_pipeline"]["accuracy"]
            ) // 2
            
            combined_syllables = []
            for i, syl in enumerate(WORD_MAP[word_id]["syllables"]):
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
            "reference_audio_url": f"/tts/generate/{word_id}",
            "detailed_results": results
        }
    
    except Exception as e:
        print(f"❌ Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/tts/generate/{word_id}")
def generate_tts_audio(word_id: str):
    """
    Generate TTS audio for a lesson
    """
    if word_id not in WORD_MAP:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    kannada_text = WORD_MAP[word_id]["text"]
    
    if not TTS_AVAILABLE or synthesizer is None:
        # Fallback: Return reference audio if available
        ref_path = f"Voices/speaker1/{int(word_id[1:])}.wav"
        if os.path.exists(ref_path):
            with open(ref_path, "rb") as f:
                return StreamingResponse(
                    io.BytesIO(f.read()),
                    media_type="audio/wav",
                    headers={"Content-Disposition": f"inline; filename={word_id}_ref.wav"}
                )
        raise HTTPException(status_code=503, detail="TTS not available")
    
    try:
        # Generate TTS
        wav = synthesizer.tts(
            text=kannada_text,
            speaker_name="male"
        )
        
        # Convert to WAV format
        byte_io = io.BytesIO()
        scipy_wav_write(byte_io, 22050, np.array(wav, dtype=np.float32))
        byte_io.seek(0)
        
        return StreamingResponse(
            byte_io,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename={word_id}_tts.wav",
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600"
            }
        )
    
    except Exception as e:
        print(f"❌ TTS Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)