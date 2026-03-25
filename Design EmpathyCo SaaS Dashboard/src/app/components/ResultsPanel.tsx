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

export function ResultsPanel() {
  const audioData = [
    { name: "Happy", value: 85, color: "#4A90E2" },
    { name: "Surprise", value: 45, color: "#95A5A6" },
    { name: "Neutral", value: 30, color: "#95A5A6" },
    { name: "Calm", value: 25, color: "#95A5A6" },
    { name: "Sad", value: 10, color: "#95A5A6" },
    { name: "Angry", value: 5, color: "#95A5A6" },
    { name: "Fear", value: 2, color: "#95A5A6" },
    { name: "Disgust", value: 1, color: "#95A5A6" },
  ];

  const textEmotions = [
    { name: "Joy", percentage: 88, color: "bg-[#2ECC71]" },
    { name: "Gratitude", percentage: 76, color: "bg-[#4A90E2]" },
    { name: "Excitement", percentage: 65, color: "bg-[#F39C12]" },
  ];

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 sm:p-8 shadow-[0_12px_40px_rgb(0,0,0,0.06)] dark:shadow-none border border-gray-50 dark:border-gray-800 transition-colors">
      <div className="flex items-start sm:items-end justify-between mb-6 flex-col sm:flex-row gap-4">
        <div>
          <h2 className="font-['Poppins'] font-bold text-3xl sm:text-4xl text-[#2C3E50] dark:text-white tracking-tight">
            Final Emotion: <br className="sm:hidden" /><span className="text-[#FF6B6B]">Happy</span>
          </h2>
          <div className="flex items-center gap-3 sm:gap-4 mt-3 sm:mt-4 w-full">
            <div className="flex-1 max-w-sm h-4 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden border border-gray-200/50 dark:border-gray-700/50 shadow-inner">
              <div
                className="h-full bg-gradient-to-r from-[#FF6B6B] to-[#ff8e8e] transition-all duration-1000 ease-out relative"
                style={{ width: "92%" }}
              >
                <div className="absolute top-0 right-0 bottom-0 left-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPSc0JyBoZWlnaHQ9JzQnPjxyZWN0IHdpZHRoPSc0JyBoZWlnaHQ9JzQnIGZpbGw9JyNmZmYnIGZpbGwtb3BhY2l0eT0nMC4yJy8+PC9zdmc+')] opacity-50"></div>
              </div>
            </div>
            <span className="font-['Inter'] font-semibold text-[#FF6B6B] text-lg">
              92%
            </span>
            <span className="font-['Inter'] text-sm text-[#7F8C8D] dark:text-gray-400">
              Confidence
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-gradient-to-r from-purple-50 dark:from-purple-900/20 to-blue-50 dark:to-blue-900/20 border border-purple-100/50 dark:border-purple-800/30 rounded-xl shadow-sm">
          <div className="w-8 h-8 rounded-full bg-white dark:bg-gray-800 shadow-sm flex items-center justify-center">
             <Link2 className="w-4 h-4 text-purple-500 dark:text-purple-400" />
          </div>
          <div>
            <p className="font-['Poppins'] font-semibold text-sm text-[#2C3E50] dark:text-gray-200 leading-none">Fusion Indicator</p>
            <p className="font-['Inter'] text-[10px] text-[#7F8C8D] dark:text-gray-400 mt-0.5 uppercase tracking-wide">Audio + Text Synergy</p>
          </div>
        </div>
      </div>

      <hr className="border-gray-100 dark:border-gray-800 mb-6" />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 sm:gap-12">
        {/* Audio Chart */}
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
                  formatter={(value: number) => [`${value}%`, "Probability"]}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16}>
                  {audioData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Text Emotions List */}
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
      </div>
    </div>
  );
}
