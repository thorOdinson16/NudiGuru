import { Routes, Route } from "react-router-dom";
import Layout from "@/Layout";

import SignIn from "@/pages/SignIn";
import Dashboard from "@/pages/Dashboard";
import Practice from "@/pages/Practice";
import Lessons from "@/pages/Lessons";
import PronounciationBattlePage from "@/pages/PronounciationBattlePage";
import Community from "@/pages/Community";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<SignIn />} />

      <Route path="/Dashboard" element={<Layout><Dashboard /></Layout>} />

      <Route path="/Practice" element={<Layout><Practice /></Layout>} />

      <Route path="/Lessons" element={<Layout><Lessons /></Layout>} />

      <Route path="/Battle" element={<Layout><PronounciationBattlePage /></Layout>} />

      <Route path="/Community" element={<Layout><Community /></Layout>} />
    </Routes>
  );
}
