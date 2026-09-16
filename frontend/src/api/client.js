import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TOKEN_KEY = "nudiguru_token";

export const api = axios.create({ baseURL: API_BASE });

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      setToken(null);
    }
    return Promise.reject(error);
  }
);

export function errorMessage(error) {
  const detail = error?.response?.data?.detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || d).join(", ");
  }
  return detail || error?.message || "Something went wrong";
}

// --- Auth ---
export const register = async (email, password, fullName) =>
  (await api.post("/auth/register", { email, password, full_name: fullName })).data;

export const login = async (email, password) =>
  (await api.post("/auth/login", { email, password })).data;

export const fetchMe = async () => (await api.get("/auth/me")).data;

// --- Lessons ---
export const fetchLessons = async () => (await api.get("/lessons")).data;

// --- User ---
export const fetchUserStats = async () => (await api.get("/user/stats")).data;

export const fetchProgress = async () => (await api.get("/user/progress")).data;

// --- Evaluation ---
export const evaluatePronunciation = async (audioBlob, lessonId) => {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");
  formData.append("lesson_id", lessonId);
  return (await api.post("/evaluate", formData)).data;
};

// --- TTS (public) ---
export const getTtsUrl = (wordId) => `${API_BASE}/tts/generate/${wordId}`;
