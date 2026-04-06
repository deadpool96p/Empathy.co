import { useState } from 'react';
import { submitFeedback } from '../services/api';

interface FeedbackUIProps {
  analysisId: string;
  inputType: 'audio' | 'text' | 'both';
  predictedEmotion: string;
  confidence: number;
  textContext: string;
}

const EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"];

export default function FeedbackUI({ analysisId, inputType, predictedEmotion, confidence, textContext }: FeedbackUIProps) {
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [feedbackType, setFeedbackType] = useState<'correct' | 'incorrect' | null>(null);
  const [correctedEmotion, setCorrectedEmotion] = useState<string>('');
  const [comment, setComment] = useState<string>('');

  const handleInitialFeedback = async (isCorrect: boolean) => {
    setFeedbackType(isCorrect ? 'correct' : 'incorrect');
    
    if (isCorrect) {
      await sendFeedback(true);
    }
  };

  const handleIncorrectSubmit = async () => {
    if (!correctedEmotion) {
      setError("Please select the correct emotion.");
      return;
    }
    await sendFeedback(false, correctedEmotion, comment);
  };

  const sendFeedback = async (isCorrect: boolean, corrected?: string, userComment?: string) => {
    setLoading(true);
    setError(null);
    try {
      await submitFeedback({
        analysis_id: analysisId,
        input_type: inputType,
        text: textContext,
        predicted_emotion: predictedEmotion,
        confidence: confidence,
        user_correct: isCorrect,
        corrected_emotion: corrected,
        comment: userComment
      });
      setSubmitted(true);
    } catch (err: any) {
      setError(err.message || "Failed to submit feedback");
      // reset if failed so they can try again
      setFeedbackType(null);
    } finally {
      setLoading(false);
    }
  };

  if (!analysisId) return null;

  if (submitted) {
    return (
      <div className="mt-6 p-4 rounded-xl border border-white/5 bg-white/5 text-center text-emerald-400">
        <p className="font-medium">Thank you for your feedback!</p>
        <p className="text-sm text-gray-400 mt-1">This will help improve our models.</p>
      </div>
    );
  }

  return (
    <div className="mt-6 p-5 rounded-xl border border-white/10 bg-white/5 backdrop-blur-sm">
      <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">Help us improve</h3>
      
      {feedbackType === null ? (
        <div className="flex items-center gap-3">
          <span className="text-gray-200">Was this prediction accurate?</span>
          <button 
            onClick={() => handleInitialFeedback(true)}
            disabled={loading}
            className="px-4 py-2 bg-white/10 hover:bg-emerald-500/20 hover:text-emerald-400 rounded-lg transition-colors disabled:opacity-50"
          >
            👍 Yes
          </button>
          <button 
            onClick={() => handleInitialFeedback(false)}
            disabled={loading}
            className="px-4 py-2 bg-white/10 hover:bg-red-500/20 hover:text-red-400 rounded-lg transition-colors disabled:opacity-50"
          >
            👎 No
          </button>
        </div>
      ) : feedbackType === 'incorrect' ? (
        <div className="space-y-4 animate-fade-in">
          <div>
            <label className="block text-sm text-gray-400 mb-1">What was the actual emotion?</label>
            <select 
              className="w-full bg-slate-800 border border-white/10 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-indigo-500"
              value={correctedEmotion}
              onChange={(e) => setCorrectedEmotion(e.target.value)}
            >
              <option value="" disabled>Select emotion...</option>
              {EMOTIONS.map(e => (
                <option key={e} value={e}>{e.charAt(0).toUpperCase() + e.slice(1)}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-gray-400 mb-1">Additional comments (optional)</label>
            <input 
              type="text"
              className="w-full bg-slate-800 border border-white/10 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-indigo-500"
              placeholder="e.g., The tone was sarcastic..."
              value={comment}
              onChange={(e) => setComment(e.target.value)}
            />
          </div>

          <div className="flex justify-end gap-2">
            <button 
              onClick={() => setFeedbackType(null)}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button 
              onClick={handleIncorrectSubmit}
              disabled={loading || !correctedEmotion}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors disabled:opacity-50"
            >
              {loading ? 'Submitting...' : 'Submit Correction'}
            </button>
          </div>
        </div>
      ) : null}
      
      {error && <p className="text-red-400 text-sm mt-3">{error}</p>}
    </div>
  );
}
