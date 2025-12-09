import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  MessageSquarePlus,
  Users,
  Calendar,
  Heart,
  Send,
  Star,
  Flame,
  Sparkles,
} from "lucide-react";

export default function Community() {
  const [posts, setPosts] = useState([
    {
      id: 1,
      user: "Ananya",
      role: "Tutor",
      content:
        "🌟 Join my FREE Kannada basics workshop this Sunday! We will cover vowels, consonants, and simple sentences.",
      likes: 12,
      comments: [{ user: "Rohit", text: "I'm joining!" }],
    },
    {
      id: 2,
      user: "Rahul",
      role: "Learner",
      content:
        "Struggling with ‘ಳ’ vs ‘ಲ’. Any simple trick to differentiate while speaking?",
      likes: 5,
      comments: [],
    },
  ]);

  const [newPost, setNewPost] = useState("");

  const addPost = () => {
    if (!newPost.trim()) return;
    setPosts([
      {
        id: Date.now(),
        user: "You",
        role: "Member",
        content: newPost,
        likes: 0,
        comments: [],
      },
      ...posts,
    ]);
    setNewPost("");
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">

      {/* --- HEADER SECTION --- */}
      <motion.div
        initial={{ opacity: 0, y: -15 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10"
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-500 bg-clip-text text-transparent mb-2">
          Kannada Learners Community
        </h1>

        <p className="text-gray-600 text-lg">
          Ask questions, join workshops, share tips, and grow your Kannada
          speaking journey with fellow learners 💬✨
        </p>
      </motion.div>

      {/* --- FEATURED WORKSHOP --- */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="mb-10 p-6 rounded-2xl shadow-xl border-0 bg-gradient-to-br from-purple-50 to-pink-50">
          <CardTitle className="text-xl font-semibold flex items-center gap-2 mb-3">
            <Sparkles className="text-purple-600" /> Featured Workshop of the Week
          </CardTitle>

          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <p className="font-medium text-gray-900 text-lg">
                🗣️ Master Kannada Retroflex Sounds – Live Class
              </p>
              <p className="text-gray-600">
                Learn tricky sounds like ಟ, ಠ, ಡ, ಢ, ಣ with live feedback.
              </p>
            </div>

            <Button className="bg-purple-600 hover:bg-purple-700 text-white flex items-center gap-2">
              <Calendar size={16} /> Join Now
            </Button>
          </div>
        </Card>
      </motion.div>

      {/* --- TRENDING TOPICS --- */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-10"
      >
        <h2 className="text-xl font-bold text-gray-800 mb-3 flex items-center gap-2">
          <Flame className="text-orange-500" /> Trending Topics
        </h2>

        <div className="flex gap-3 flex-wrap">
          {[
            "Pronunciation Tips",
            "Retroflex Letters",
            "Daily Kannada Phrases",
            "Beginner Doubts",
            "Speaking Practice",
          ].map((tag) => (
            <span
              key={tag}
              className="px-4 py-2 bg-orange-100 text-orange-700 rounded-full text-sm font-medium hover:bg-orange-200 transition cursor-pointer"
            >
              #{tag}
            </span>
          ))}
        </div>
      </motion.div>

      {/* --- TOP TUTORS SECTION --- */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.25 }}
        className="mb-12"
      >
        <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <Star className="text-yellow-500" /> Top Tutors This Week
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {["Ananya", "Charan", "Deepika"].map((tutor, idx) => (
            <motion.div
              key={tutor}
              whileHover={{ scale: 1.05 }}
              className="p-5 rounded-2xl border shadow-md bg-white"
            >
              <h3 className="font-semibold text-gray-900">{tutor}</h3>
              <p className="text-gray-500 text-sm">Certified Kannada Tutor</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* --- NEW POST INPUT --- */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <Card className="p-4 shadow-md rounded-2xl">
          <textarea
            className="w-full p-3 rounded-lg border focus:ring-2 focus:ring-purple-400 outline-none"
            placeholder="Share something with the community..."
            value={newPost}
            onChange={(e) => setNewPost(e.target.value)}
          />

          <div className="flex justify-end mt-3">
            <Button
              onClick={addPost}
              className="bg-purple-600 hover:bg-purple-700 text-white flex items-center gap-2"
            >
              <MessageSquarePlus size={16} /> Post
            </Button>
          </div>
        </Card>
      </motion.div>

      {/* --- POSTS --- */}
      <div className="space-y-6">
        {posts.map((post) => (
          <motion.div
            key={post.id}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card className="p-5 rounded-2xl shadow-lg border border-gray-200">
              <CardHeader className="pb-2">
                <div className="flex justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900">{post.user}</h3>
                    <p className="text-sm text-gray-500">{post.role}</p>
                  </div>
                </div>
              </CardHeader>

              <CardContent>
                <p className="mb-4 text-gray-800">{post.content}</p>

                <div className="flex items-center gap-6 text-gray-600">
                  <button className="flex items-center gap-1 hover:text-red-500 transition">
                    <Heart className="w-4 h-4" /> {post.likes}
                  </button>

                  <button className="flex items-center gap-1 hover:text-purple-600 transition">
                    <Users className="w-4 h-4" /> {post.comments.length} comments
                  </button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  );
}