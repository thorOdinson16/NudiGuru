import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { CheckCircle2, Circle, Target, Play, Lock } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fetchLessons, fetchProgress } from "@/api/client";

export default function Lessons() {
  const { data: lessons = [], isLoading } = useQuery({
    queryKey: ["lessons"],
    queryFn: fetchLessons,
  });

  const { data: progress = [] } = useQuery({
    queryKey: ["progress"],
    queryFn: fetchProgress,
  });

  const getLessonStatus = (lessonId) => {
    const entry = progress.find((p) => p.lesson_id === lessonId);
    if (!entry) return "not_attempted";
    if (entry.best_accuracy >= 90) return "perfected";
    if (entry.best_accuracy >= 75) return "practiced";
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
                <Link to={`/Practice?lesson=${lesson.id}`}>
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
