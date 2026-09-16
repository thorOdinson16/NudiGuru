import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";

import { useAuth } from "@/contexts/AuthContext";
import { errorMessage } from "@/api/client";

export default function SignUp() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const { register, user } = useAuth();
  const navigate = useNavigate();

  if (user) {
    return <Navigate to="/Dashboard" replace />;
  }

  const handleRegister = async (event) => {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await register(email, password, fullName);
      navigate("/Dashboard", { replace: true });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-200 to-emerald-300 p-6">
      <form
        onSubmit={handleRegister}
        className="bg-white/80 backdrop-blur-xl p-8 rounded-2xl shadow-xl w-full max-w-md"
      >
        <h1 className="text-3xl font-bold text-center mb-6">Create Account ✨</h1>

        {error && (
          <p className="mb-4 rounded-xl bg-red-50 px-4 py-2 text-sm text-red-700">
            {error}
          </p>
        )}

        <input
          type="text"
          placeholder="Full Name"
          className="w-full p-3 border rounded-xl mb-4 focus:ring-2 focus:ring-green-400"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
        />

        <input
          type="email"
          required
          placeholder="Enter Email"
          className="w-full p-3 border rounded-xl mb-4 focus:ring-2 focus:ring-green-400"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          type="password"
          required
          minLength={6}
          placeholder="Password (min 6 characters)"
          className="w-full p-3 border rounded-xl mb-4 focus:ring-2 focus:ring-green-400"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button
          type="submit"
          disabled={submitting}
          className="w-full bg-green-500 hover:bg-green-600 disabled:opacity-60 text-white p-3 rounded-xl font-semibold"
        >
          {submitting ? "Creating account…" : "Sign Up"}
        </button>

        <p className="mt-4 text-center text-gray-700">
          Already have an account?{" "}
          <Link to="/signin" className="text-green-600 font-semibold">
            Sign In
          </Link>
        </p>
      </form>
    </div>
  );
}
