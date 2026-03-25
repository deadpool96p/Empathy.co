import { Upload, Mic, Play, CheckCircle } from "lucide-react";

export function AudioInputCard() {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 sm:p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:shadow-none h-full flex flex-col border border-gray-50 dark:border-gray-800 transition-colors">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-['Poppins'] font-semibold text-lg text-[#2C3E50] dark:text-white">
          Audio Input
        </h3>
        <span className="text-xs font-['Inter'] font-medium px-2.5 py-1 bg-blue-50 dark:bg-blue-900/30 text-[#4A90E2] dark:text-blue-400 rounded-full">
          Primary
        </span>
      </div>

      <div className="border-2 border-dashed border-gray-200 dark:border-gray-700 hover:border-[#4A90E2] dark:hover:border-[#4A90E2] transition-colors rounded-xl p-6 sm:p-8 flex flex-col items-center justify-center flex-1 bg-gray-50/50 dark:bg-gray-800/50 cursor-pointer group">
        <div className="w-12 h-12 rounded-full bg-blue-50 dark:bg-blue-900/30 group-hover:bg-blue-100 dark:group-hover:bg-blue-800/40 flex items-center justify-center mb-4 transition-colors">
          <Upload className="w-6 h-6 text-[#4A90E2]" />
        </div>
        <p className="font-['Inter'] text-sm font-medium text-[#2C3E50] dark:text-gray-200 text-center mb-1">
          Drag & drop audio file
        </p>
        <p className="font-['Inter'] text-xs text-[#7F8C8D] dark:text-gray-400 text-center">
          WAV, MP3, FLAC (Max 25MB)
        </p>
      </div>

      <div className="mt-5 flex items-center gap-3">
        <button className="flex items-center justify-center gap-2 flex-1 py-2.5 bg-[#F5F7FA] dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-[#2C3E50] dark:text-white rounded-xl font-['Inter'] text-sm font-medium transition-colors border border-gray-100 dark:border-gray-700">
          <Mic className="w-4 h-4 text-[#FF6B6B]" />
          Record
        </button>
        <button className="w-10 h-10 flex items-center justify-center bg-[#F5F7FA] dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-xl text-[#2C3E50] dark:text-white transition-colors border border-gray-100 dark:border-gray-700">
          <Play className="w-4 h-4" />
        </button>
      </div>

      {/* Visualizer & File info */}
      <div className="mt-5 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-100 dark:border-gray-700 flex items-center gap-4">
        <div className="flex-1 flex items-center gap-1 sm:gap-1.5 h-8 overflow-hidden">
          {[...Array(30)].map((_, i) => (
            <div
              key={i}
              className="w-1.5 bg-[#4A90E2] rounded-full opacity-60"
              style={{
                height: `${Math.max(20, Math.random() * 100)}%`,
                opacity: i % 2 === 0 ? 0.8 : 0.4,
              }}
            ></div>
          ))}
        </div>
        <div className="flex flex-col items-end flex-shrink-0">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-xs text-[#2C3E50] dark:text-gray-200 font-medium">customer_call_01.wav</span>
            <CheckCircle className="w-3.5 h-3.5 text-[#2ECC71]" />
          </div>
          <span className="font-['Inter'] text-[10px] text-[#7F8C8D] dark:text-gray-400 mt-0.5">02:45 • 4.2 MB</span>
        </div>
      </div>
    </div>
  );
}
