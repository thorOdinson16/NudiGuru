import { NavLink, useNavigate } from "react-router-dom";
import { Home, Mic, Library, Users, Swords, LogOut } from "lucide-react";

import { useAuth } from "@/contexts/AuthContext";

const navItems = [
  { to: "/Dashboard", label: "Dashboard", icon: Home, active: "from-yellow-400 to-orange-500" },
  { to: "/Practice", label: "Practice", icon: Mic, active: "from-orange-500 to-pink-500" },
  { to: "/Lessons", label: "Lessons", icon: Library, active: "from-blue-500 to-indigo-500" },
  { to: "/Battle", label: "Battle", icon: Swords, active: "from-red-500 to-orange-500" },
  { to: "/Community", label: "Community", icon: Users, active: "from-purple-500 to-pink-500" },
];

export default function Header() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/signin", { replace: true });
  };

  return (
    <header className="w-full bg-white/80 backdrop-blur-md shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">
        {/* Logo */}
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
        <nav className="flex items-center gap-4">
          {navItems.map(({ to, label, icon: Icon, active }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-xl font-medium transition ${
                  isActive
                    ? `bg-gradient-to-r ${active} text-white shadow-md`
                    : "text-gray-700 hover:bg-gray-100"
                }`
              }
            >
              <Icon size={18} /> {label}
            </NavLink>
          ))}

          {user && (
            <div className="flex items-center gap-3 pl-4 ml-2 border-l border-gray-200">
              <span className="hidden sm:block text-sm font-medium text-gray-700">
                {user.full_name || user.email}
              </span>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1 px-3 py-2 rounded-xl text-sm font-medium text-red-600 hover:bg-red-50 transition"
              >
                <LogOut size={16} /> Logout
              </button>
            </div>
          )}
        </nav>
      </div>
    </header>
  );
}
