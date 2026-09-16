import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

import {
  Mic,
  Library,
  Flame,
  Target,
  Award,
  ChevronRight,
  Swords,
  Users,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ProgressRing from "@/components/ui/ProgressRing";
import BottomStats from "@/components/ui/BottomStats";
import { fetchUserStats } from "@/api/client";

export default function Dashboard() {
  const [greeting, setGreeting] = useState("");

  const { data: user, isLoading } = useQuery({
    queryKey: ["userStats"],
    queryFn: fetchUserStats,
  });

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting("ಶುಭೋದಯ");
    else if (hour < 17) setGreeting("ಶುಭ ಮಧ್ಯಾಹ್ನ");
    else setGreeting("ಶುಭ ಸಂಜೆ");
  }, []);

  const dailyProgress = user?.daily_practice_count || 0;
  const dailyGoal = user?.daily_goal || 10;
  const streak = user?.streak_days || 0;
  const avgAccuracy = user?.avg_accuracy || 0;

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500" />
      </div>
    );
  }

  return (
    <>
      <div className="min-h-screen pb-40 bg-gradient-to-br from-orange-100 via-white to-sky-200 animate-gradient-slow transition-all">
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Greeting */}
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-2 kannada-text">
              {greeting}, {user?.full_name || "Friend"}!
            </h1>
            <p className="text-lg text-gray-600">
              Ready to practice your Kannada pronunciation today?
            </p>
          </motion.div>

          {/* Main Stats */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid md:grid-cols-3 gap-6 mb-12"
          >
            <motion.div variants={itemVariants}>
              <Card className="glass-card border-0 shadow-lg hover:shadow-xl">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5 text-orange-500" />
                    ಇಂದಿನ ಪ್ರಗತಿ / Today's Progress
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col items-center pt-2">
                  <ProgressRing progress={dailyProgress} total={dailyGoal} size={140} color="url(#gradient1)" />
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

            <motion.div variants={itemVariants}>
              <Card className="glass-card border-0 shadow-lg hover:shadow-xl">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Flame className="w-5 h-5 text-orange-500" />
                    ಪ್ರಸ್ತುತ / Current Streak
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col items-center">
                  <div className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-500 to-pink-500">
                    {streak}
                  </div>
                  <p className="text-sm text-gray-600 mt-2">days in a row</p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div variants={itemVariants}>
              <Card className="glass-card border-0 shadow-lg hover:shadow-xl">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Award className="w-5 h-5 text-blue-500" />
                    ಸರಾಸರಿ ನಿಖರತೆ / Average Accuracy
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col items-center">
                  <div className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-cyan-500">
                    {avgAccuracy}%
                  </div>
                  <p className="text-sm text-gray-600 mt-2">across all practices</p>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>

          {/* JOURNEY SECTION */}
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
            <motion.div variants={itemVariants}>
              <Link to="/Practice">
                <Card className="glass-card border-0 shadow-lg hover:shadow-2xl transition-all hover:scale-105 cursor-pointer group h-full relative overflow-hidden">
                  <CardContent className="p-8 flex flex-col items-center text-center">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-orange-400 to-pink-500 flex items-center justify-center mb-4">
                      <Mic className="w-10 h-10 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">ಈಗ ಕಲಿಯೋಣ / Practice Now</h3>
                    <p className="text-gray-600 mb-4">
                      Start practicing your pronunciation with instant feedback
                    </p>
                    <ChevronRight className="w-6 h-6 text-gray-400 group-hover:text-orange-500" />
                  </CardContent>
                </Card>
              </Link>
            </motion.div>

            <motion.div variants={itemVariants}>
              <Link to="/Lessons">
                <Card className="glass-card border-0 shadow-lg hover:shadow-2xl transition-all hover:scale-105 cursor-pointer group h-full relative overflow-hidden">
                  <CardContent className="p-8 flex flex-col items-center text-center">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center mb-4">
                      <Library className="w-10 h-10 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">ಪಾಠಗಳು / Lessons</h3>
                    <p className="text-gray-600 mb-4">
                      Browse benchmark lessons at your own pace
                    </p>
                    <ChevronRight className="w-6 h-6 text-gray-400 group-hover:text-green-500" />
                  </CardContent>
                </Card>
              </Link>
            </motion.div>

            <motion.div variants={itemVariants}>
              <Link to="/Battle">
                <Card className="glass-card border-0 shadow-lg hover:shadow-2xl transition-all hover:scale-105 cursor-pointer group h-full relative overflow-hidden">
                  <CardContent className="p-8 flex flex-col items-center text-center">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-purple-500 to-indigo-500 flex items-center justify-center mb-4 shadow-lg">
                      <Swords className="w-10 h-10 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      ಉಚ್ಚಾರಣೆ ಸಮರ / Pronunciation Battle
                    </h3>
                    <p className="text-gray-600 mb-4">
                      Compete with a friend and see who pronounces better
                    </p>
                    <ChevronRight className="w-6 h-6 text-gray-400 group-hover:text-purple-500" />
                  </CardContent>
                </Card>
              </Link>
            </motion.div>

            <motion.div variants={itemVariants}>
              <Link to="/Community">
                <Card className="glass-card border-0 shadow-lg hover:shadow-2xl transition-all hover:scale-105 cursor-pointer group h-full relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-pink-500/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                  <CardContent className="p-8 flex flex-col items-center text-center relative z-10">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center mb-4 shadow-lg">
                      <Users className="w-10 h-10 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">ಬಳಗ / Community</h3>
                    <p className="text-gray-600 mb-4">
                      Join discussions, workshops & meet other learners
                    </p>
                    <ChevronRight className="w-6 h-6 text-gray-400 group-hover:text-purple-500" />
                  </CardContent>
                </Card>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </div>

      <BottomStats
        userStats={{
          totalPractices: user?.total_practices || 0,
          uniqueLessons: user?.unique_lessons || 0,
          avgAccuracy: avgAccuracy,
          streak: streak,
        }}
      />
    </>
  );
}
