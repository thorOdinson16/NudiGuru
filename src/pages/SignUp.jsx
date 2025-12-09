import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function SignUp() {
  const [email, setEmail] = useState("");
  const navigate = useNavigate();

  const handleRegister = () => {
    // Dummy signup → treat as logged in
    localStorage.setItem("nudiguru_user", email);
    navigate("/Dashboard");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-200 to-emerald-300 p-6">
      <div className="bg-white/80 backdrop-blur-xl p-8 rounded-2xl shadow-xl w-full max-w-md">

        <h1 className="text-3xl font-bold text-center mb-6">Create Account ✨</h1>

        <input
          type="email"
          placeholder="Enter Email"
          className="w-full p-3 border rounded-xl mb-4 focus:ring-2 focus:ring-green-400"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <button
          onClick={handleRegister}
          className="w-full bg-green-500 hover:bg-green-600 text-white p-3 rounded-xl font-semibold"
        >
          Sign Up
        </button>

        <p className="mt-4 text-center text-gray-700">
          Already have an account?{" "}
          <Link to="/signin" className="text-green-600 font-semibold">
            Sign In
          </Link>
        </p>

      </div>
    </div>
  );
}
