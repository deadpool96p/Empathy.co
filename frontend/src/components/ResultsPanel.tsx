import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";
import { Link2 } from "lucide-react";
import type { AnalysisResponse } from "../services/types";
import FeedbackUI from "./FeedbackUI";

interface ResultsPanelProps {
  results: AnalysisResponse;
}

export function ResultsPanel({ results }: ResultsPanelProps) {
  // Extract probabilities or use defaults
  const audioData = results.audio?.probabilities 
    ? [
        { name: "Neutral", value: Math.round(results.audio.probabilities[0] * 100), color: "#95A5A6" },
        { name: "Calm", value: Math.round(results.audio.probabilities[1] * 100), color: "#95A5A6" },
        { name: "Happy", value: Math.round(results.audio.probabilities[2] * 100), color: "#4A90E2" },
        { name: "Sad", value: Math.round(results.audio.probabilities[3] * 100), color: "#95A5A6" },
        { name: "Angry", value: Math.round(results.audio.probabilities[4] * 100), color: "#E74C3C" },
        { name: "Fear", value: Math.round(results.audio.probabilities[5] * 100), color: "#95A5A6" },
        { name: "Disgust", value: Math.round(results.audio.probabilities[6] * 100), color: "#95A5A6" },
        { name: "Surprise", value: Math.round(results.audio.probabilities[7] * 100), color: "#95A5A6" },
      ].sort((a, b) => b.value - a.value)
    : [];

  const textEmotions = results.text?.top_3 
    ? results.text.top_3.map((t, idx) => ({
        name: t.emotion.charAt(0).toUpperCase() + t.emotion.slice(1),
        percentage: Math.round(t.score * 100),
        color: idx === 0 ? "bg-[#2ECC71]" : idx === 1 ? "bg-[#4A90E2]" : "bg-[#F39C12]"
      }))
    : [];

  const finalEmotionStr = results.final_emotion 
    ? results.final_emotion.charAt(0).toUpperCase() + results.final_emotion.slice(1)
    : "Unknown";

  const finalConfidencePct = results.final_confidence 
    ? Math.round(results.final_confidence * 100)
    : 0;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 sm:p-8 shadow-[0_12px_40px_rgb(0,0,0,0.06)] dark:shadow-none border border-gray-50 dark:border-gray-800 transition-colors animate-in fade-in zoom-in duration-500">
      <div className="flex items-start sm:items-end justify-between mb-6 flex-col sm:flex-row gap-4">
        <div className="w-full">
          <h2 className="font-['Poppins'] font-bold text-3xl sm:text-4xl text-[#2C3E50] dark:text-white tracking-tight">
            Final Emotion: <br className="sm:hidden" /><span className="text-[#FF6B6B]">{finalEmotionStr}</span>
          </h2>
          <div className="flex items-center gap-3 sm:gap-4 mt-3 sm:mt-4 w-full">
            <div className="flex-1 max-w-sm h-4 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden border border-gray-200/50 dark:border-gray-700/50 shadow-inner">
              <div
                className="h-full bg-gradient-to-r from-[#FF6B6B] to-[#ff8e8e] transition-all duration-1000 ease-out relative"
                style={{ width: `${finalConfidencePct}%` }}
              >
                <div className="absolute top-0 right-0 bottom-0 left-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPSc0JyBoZWlnaHQ9JzQnPjxyZWN0IHdpZHRoPSc0JyBoZWlnaHQ9JzQnIGZpbGw9JyNmZmYnIGZpbGwtb3BhY2l0eT0nMC4yJy8+PC9zdmc+')] opacity-50"></div>
              </div>
            </div>
            <span className="font-['Inter'] font-semibold text-[#FF6B6B] text-lg">
              {finalConfidencePct}%
            </span>
            <span className="font-['Inter'] text-sm text-[#7F8C8D] dark:text-gray-400">
              Confidence
            </span>
          </div>
        </div>

        {(results.audio && results.text) && (
          <div className="flex items-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-gradient-to-r from-purple-50 dark:from-purple-900/20 to-blue-50 dark:to-blue-900/20 border border-purple-100/50 dark:border-purple-800/30 rounded-xl shadow-sm flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-white dark:bg-gray-800 shadow-sm flex items-center justify-center">
               <Link2 className="w-4 h-4 text-purple-500 dark:text-purple-400" />
            </div>
            <div>
              <p className="font-['Poppins'] font-semibold text-sm text-[#2C3E50] dark:text-gray-200 leading-none">Fusion Indicator</p>
              <p className="font-['Inter'] text-[10px] text-[#7F8C8D] dark:text-gray-400 mt-0.5 uppercase tracking-wide">Audio + Text Synergy</p>
            </div>
          </div>
        )}
      </div>

      <hr className="border-gray-100 dark:border-gray-800 mb-6" />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 sm:gap-12">
        {/* Audio Chart */}
        {results.audio ? (
          <div>
            <h4 className="font-['Poppins'] font-semibold text-[#2C3E50] dark:text-white mb-6 flex items-center gap-2">
               <span className="w-2 h-2 rounded-full bg-[#4A90E2]"></span>
               Audio Class Probabilities
            </h4>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={audioData}
                  layout="vertical"
                  margin={{ top: 0, right: 30, left: 20, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E5E7EB" />
                  <XAxis type="number" hide domain={[0, 100]} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontFamily: "Inter", fontSize: 13, fill: "#7F8C8D", fontWeight: 500 }}
                    width={80}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(150,150,150,0.1)" }}
                    contentStyle={{ borderRadius: "8px", border: "none", boxShadow: "0 4px 20px rgba(0,0,0,0.08)", fontFamily: "Inter", fontWeight: 500 }}
                    formatter={(value: any) => [`${value}%`, "Probability"]}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16}>
                    {audioData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={index === 0 ? "#4A90E2" : entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full min-h-[200px] border border-dashed border-gray-200 dark:border-gray-800 rounded-xl bg-gray-50/50 dark:bg-gray-800/30">
            <p className="text-sm font-['Inter'] text-gray-400 dark:text-gray-500">No audio analysis available</p>
          </div>
        )}

        {/* Text Emotions List */}
        {results.text ? (
          <div>
            <h4 className="font-['Poppins'] font-semibold text-[#2C3E50] dark:text-white mb-6 flex items-center gap-2">
               <span className="w-2 h-2 rounded-full bg-[#2ECC71]"></span>
               Top 3 Text Emotions
            </h4>
            <div className="flex flex-col gap-4">
              {textEmotions.map((emotion, idx) => (
                <div
                  key={idx}
                  className="group relative bg-[#F5F7FA] dark:bg-gray-800 hover:bg-white dark:hover:bg-gray-750 border border-transparent hover:border-gray-200 dark:hover:border-gray-600 rounded-xl p-5 transition-all shadow-sm hover:shadow-md flex items-center justify-between overflow-hidden"
                >
                  {/* Background progress bar effect */}
                  <div 
                    className={`absolute left-0 top-0 bottom-0 opacity-10 dark:opacity-20 ${emotion.color} transition-all duration-500 ease-in-out`} 
                    style={{ width: `${emotion.percentage}%` }}
                  ></div>
                  
                  <div className="relative z-10 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-white dark:bg-gray-900 shadow-sm flex items-center justify-center font-['Poppins'] font-semibold text-gray-400 dark:text-gray-500 text-lg">
                      #{idx + 1}
                    </div>
                    <span className="font-['Inter'] font-medium text-lg text-[#2C3E50] dark:text-white">
                      {emotion.name}
                    </span>
                  </div>
                  
                  <div className="relative z-10 flex flex-col items-end">
                    <span className="font-['Poppins'] font-semibold text-xl text-[#2C3E50] dark:text-white">
                      {emotion.percentage}%
                    </span>
                    <span className="font-['Inter'] text-xs text-[#7F8C8D] dark:text-gray-400">Score</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full min-h-[200px] border border-dashed border-gray-200 dark:border-gray-800 rounded-xl bg-gray-50/50 dark:bg-gray-800/30">
            <p className="text-sm font-['Inter'] text-gray-400 dark:text-gray-500">No text analysis available</p>
          </div>
        )}
      </div>

      {results.summary && (
        <div className="mt-8 p-6 bg-gradient-to-br from-gray-50 to-white dark:from-gray-800/50 dark:to-gray-900/50 border border-gray-100 dark:border-gray-800 rounded-2xl shadow-sm relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1 h-full bg-[#FF6B6B]"></div>
          <h4 className="font-['Poppins'] font-bold text-[#2C3E50] dark:text-white mb-3 flex items-center gap-2">
            <span className="p-1.5 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <svg className="w-4 h-4 text-[#FF6B6B]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </span>
            Emotion Summary & Insights
          </h4>
          <p className="font-['Inter'] text-[#34495E] dark:text-gray-300 leading-relaxed text-lg italic">
            "{results.summary}"
          </p>
          {results.detected_language && results.detected_language !== 'en' && (
            <p className="mt-3 font-['Inter'] text-[10px] text-[#7F8C8D] dark:text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
              <span className="w-1 h-1 rounded-full bg-blue-400"></span>
              Translated from {results.detected_language === 'hi' ? 'Hindi' : results.detected_language === 'mr' ? 'Marathi' : results.detected_language}
            </p>
          )}
        </div>
      )}

      {results.analysis_id && (
        <FeedbackUI 
          analysisId={results.analysis_id} 
          inputType={results.audio && results.text ? 'both' : results.audio ? 'audio' : 'text'}
          predictedEmotion={results.final_emotion || 'neutral'}
          confidence={results.final_confidence || 0.5}
          textContext={results.summary || ''}
        />
      )}
    </div>
  );
}

