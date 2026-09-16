import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "@/Layout";
import ProtectedRoute from "@/components/ui/ProtectedRoute";

import SignIn from "@/pages/SignIn";
import SignUp from "@/pages/SignUp";
import Dashboard from "@/pages/Dashboard";
import Practice from "@/pages/Practice";
import Lessons from "@/pages/Lessons";
import PronounciationBattlePage from "@/pages/PronounciationBattlePage";
import Community from "@/pages/Community";

function Protected({ children }) {
  return (
    <ProtectedRoute>
      <Layout>{children}</Layout>
    </ProtectedRoute>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/Dashboard" replace />} />
      <Route path="/signin" element={<SignIn />} />
      <Route path="/signup" element={<SignUp />} />

      <Route path="/Dashboard" element={<Protected><Dashboard /></Protected>} />
      <Route path="/Practice" element={<Protected><Practice /></Protected>} />
      <Route path="/Lessons" element={<Protected><Lessons /></Protected>} />
      <Route path="/Battle" element={<Protected><PronounciationBattlePage /></Protected>} />
      <Route path="/Community" element={<Protected><Community /></Protected>} />

      <Route path="*" element={<Navigate to="/Dashboard" replace />} />
    </Routes>
  );
}
