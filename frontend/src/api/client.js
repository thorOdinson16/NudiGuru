import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE,
});

// Lessons
export const fetchLessons = async () => {
  const { data } = await api.get('/lessons');
  return data;
};

// User stats
export const fetchUserStats = async () => {
  const { data } = await api.get('/user/stats');
  return data;
};

// Evaluate pronunciation
export const evaluatePronunciation = async (audioBlob, wordId) => {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');
  formData.append('word_id', wordId);
  
  const { data } = await api.post('/evaluate', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  
  return data;
};

// Get native audio URL
export const getNativeAudioUrl = (wordId) => {
  return `${API_BASE}/tts/${wordId}`;
};