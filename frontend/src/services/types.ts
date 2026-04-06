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
  analysis_id?: string;
  final_emotion?: string;
  final_confidence?: number;
  audio?: AudioAnalysis;
  text?: TextAnalysis;
  summary?: string;
  detected_language?: string;
  error?: string;
}

export interface TranscriptionResponse {
  text: string;
  error?: string;
}

export interface ModelsResponse {
  models: string[];
}

export interface FeedbackSubmission {
  analysis_id: string;
  input_type: 'audio' | 'text' | 'both';
  text?: string;
  predicted_emotion: string;
  confidence: number;
  user_correct: boolean;
  corrected_emotion?: string;
  comment?: string;
}
