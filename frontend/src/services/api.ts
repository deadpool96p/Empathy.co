import axios from 'axios';
import type { AnalysisResponse, TranscriptionResponse, ModelsResponse } from './types';

// Use env var or default to relative path string which Vite proxy handles
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/**
 * Mocking the API responses for development as requested.
 * In a real scenario, this would post FormData to the actual backend.
 */
export const analyze = async (audio: File | null, text: string, language: string): Promise<AnalysisResponse> => {
  // If both are missing, throw error
  if (!audio && !text.trim()) {
    throw new Error("Please provide either audio or text for analysis.");
  }

  const formData = new FormData();
  if (audio) {
    formData.append('audio', audio);
  }
  if (text.trim()) {
    formData.append('text', text.trim());
  }
  if (language) {
    formData.append('language', language);
  }

  try {
    const response = await api.post<AnalysisResponse>('/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(typeof error.response.data.detail === 'string' ? error.response.data.detail : JSON.stringify(error.response.data.detail));
    }
    throw new Error(error.message || "Failed to analyze input.");
  }
};

export const transcribe = async (audio: File): Promise<TranscriptionResponse> => {
  const formData = new FormData();
  formData.append('audio', audio);

  try {
    const response = await api.post<TranscriptionResponse>('/transcribe', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(typeof error.response.data.detail === 'string' ? error.response.data.detail : JSON.stringify(error.response.data.detail));
    }
    throw new Error(error.message || "Failed to transcribe audio.");
  }
};

export const getModels = async (): Promise<ModelsResponse> => {
  try {
    const response = await api.get<ModelsResponse>('/models');
    return response.data;
  } catch (error: any) {
    console.error("Failed to load models list", error);
    throw new Error(error.message || "Failed to load models list");
  }
};
