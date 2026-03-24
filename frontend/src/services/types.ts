export interface EmotionPrediction {
  emotion: string;
  confidence: number;
}

export interface AudioAnalysis {
  emotion: string;
  confidence: number;
  probabilities: number[]; // length 8
}

export interface TextEmotionScore {
  emotion: string;
  score: number;
}

export interface TextAnalysis {
  emotion: string;
  confidence: number;
  top_3: TextEmotionScore[];
}

export interface AnalysisResponse {
  final_emotion?: string;
  final_confidence?: number;
  audio?: AudioAnalysis;
  text?: TextAnalysis;
  error?: string;
}

export interface TranscriptionResponse {
  text: string;
  error?: string;
}

export interface ModelsResponse {
  models: string[];
}
