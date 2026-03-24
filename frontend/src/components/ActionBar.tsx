import { Globe, Sparkles, ChevronDown, Loader2 } from "lucide-react";
import * as Select from "@radix-ui/react-select";
import { motion } from "motion/react";

interface ActionBarProps {
  language: string;
  setLanguage: (lang: string) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  canAnalyze: boolean;
  models: string[];
}

export function ActionBar({ language, setLanguage, onAnalyze, isAnalyzing, canAnalyze, models }: ActionBarProps) {
  const languageLabels: Record<string, string> = {
    multi: "Multilingual",
    en: "English",
    mr: "Marathi",
    hi: "Hindi"
  };

  return (
    <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3 my-4 py-4 px-4 bg-white/60 dark:bg-gray-900/60 backdrop-blur-md rounded-2xl border border-white/40 dark:border-gray-800/40 shadow-[0_4px_30px_rgba(0,0,0,0.03)] dark:shadow-[0_4px_30px_rgba(0,0,0,0.4)] w-full transition-colors relative z-10">
      <div className="flex flex-col sm:flex-row gap-3 w-full max-w-[600px] mx-auto">
        
        {/* Language Selector */}
        <Select.Root value={language} onValueChange={setLanguage}>
          <Select.Trigger className="flex-1 min-w-[140px] flex justify-between items-center gap-2 px-4 py-3 sm:py-3.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 rounded-xl text-[#2C3E50] dark:text-gray-200 font-['Inter'] text-sm font-medium transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-100 dark:focus:ring-blue-900">
            <div className="flex items-center gap-2">
              <Globe className="w-4 h-4 text-[#7F8C8D] dark:text-gray-400" />
              <Select.Value />
            </div>
            <Select.Icon>
              <ChevronDown className="w-4 h-4 text-[#7F8C8D] dark:text-gray-400" />
            </Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Content className="overflow-hidden bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-100 dark:border-gray-700 z-50">
              <Select.Viewport className="p-1">
                {models.map((model) => (
                  <Select.Item
                    key={model}
                    value={model}
                    className="flex items-center gap-2 px-6 py-3 text-sm font-['Inter'] font-medium text-[#2C3E50] dark:text-gray-200 outline-none cursor-pointer data-[highlighted]:bg-gray-50 dark:data-[highlighted]:bg-gray-700 rounded-lg transition-colors data-[state=checked]:text-[#4A90E2] data-[state=checked]:bg-blue-50 dark:data-[state=checked]:bg-blue-900/30"
                  >
                    <Select.ItemText>{languageLabels[model] || model}</Select.ItemText>
                  </Select.Item>
                ))}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>

        {/* Action Button */}
        <motion.button 
          whileTap={canAnalyze && !isAnalyzing ? { scale: 0.98 } : {}}
          onClick={onAnalyze}
          disabled={!canAnalyze || isAnalyzing}
          className={`flex-2 flex items-center justify-center gap-2 px-8 py-3.5 rounded-xl font-['Poppins'] font-semibold text-sm sm:text-base transition-all shadow-[0_8px_20px_rgba(74,144,226,0.3)] hover:shadow-[0_8px_25px_rgba(74,144,226,0.4)] ${
            canAnalyze && !isAnalyzing
              ? "bg-[#4A90E2] hover:bg-[#357ABD] active:bg-[#2C6EAF] text-white"
              : "bg-blue-300 dark:bg-blue-900/50 text-blue-100 dark:text-blue-300 cursor-not-allowed shadow-none hover:shadow-none"
          }`}
        >
          {isAnalyzing ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Sparkles className={`w-5 h-5 ${canAnalyze ? 'animate-pulse' : ''}`} />
          )}
          {isAnalyzing ? 'Analyzing...' : 'Analyze Emotion'}
        </motion.button>
      </div>
    </div>
  );
}
