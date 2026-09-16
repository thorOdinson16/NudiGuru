import React from "react";
import { NavLink } from "react-router-dom";
import { Home, Mic, Library, Users } from "lucide-react";
import { Swords } from "lucide-react";


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

          <NavLink
            to="/Battle"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition 
              ${isActive ? "bg-gradient-to-r from-red-500 to-orange-500 text-white shadow-md" : "text-gray-700"}`
            }
          >
            <Swords size={18} /> Battle
          </NavLink>
          
          <NavLink
            to="/Community"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition 
              ${isActive ? "bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-md" : "text-gray-700"}`
            }
          >
            <Users size={18} /> Community
          </NavLink>        
        </nav>
      </div>
    </header>
  );
}
