import { FileText, Cpu, Loader2 } from "lucide-react";
import { useState } from "react";
import { transcribe } from "../services/api";
import { toast } from "react-toastify";

interface TextInputProps {
  text: string;
  setText: (t: string) => void;
  audioFile: File | null;
}

export function TextInput({ text, setText, audioFile }: TextInputProps) {
  const [isTranscribing, setIsTranscribing] = useState(false);

  const handleAutoTranscribe = async () => {
    if (!audioFile) return;
    
    setIsTranscribing(true);
    try {
      const response = await transcribe(audioFile);
      setText(response.text);
      toast.success("Audio transcribed successfully");
    } catch (err: any) {
      toast.error(err.message || "Failed to transcribe audio");
    } finally {
      setIsTranscribing(false);
    }
  };

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const charCount = text.length;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl p-5 sm:p-6 shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:shadow-none h-full flex flex-col border border-gray-50 dark:border-gray-800 relative group transition-colors">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-['Poppins'] font-semibold text-lg text-[#2C3E50] dark:text-white">
          Text Input
        </h3>
        <span className="text-xs font-['Inter'] font-medium px-2.5 py-1 bg-gray-50 dark:bg-gray-800 text-[#7F8C8D] dark:text-gray-400 rounded-full">
          Secondary
        </span>
      </div>

      <div className="relative flex-1 flex flex-col">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-full flex-1 min-h-[160px] p-4 sm:p-5 bg-[#F5F7FA] dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 focus:border-[#4A90E2] dark:focus:border-[#4A90E2] focus:ring-1 focus:ring-[#4A90E2] outline-none resize-none font-['Inter'] text-sm sm:text-base text-[#2C3E50] dark:text-gray-200 leading-relaxed transition-colors group-hover:bg-white dark:group-hover:bg-gray-800"
          placeholder="Enter transcript manually or auto-transcribe from audio..."
        ></textarea>
        
        <div className="absolute bottom-4 right-4">
          <button 
            onClick={handleAutoTranscribe}
            disabled={!audioFile || isTranscribing}
            className={`flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-700 border shadow-sm rounded-lg font-['Inter'] text-sm font-medium transition-all ${
              !audioFile || isTranscribing 
                ? "border-gray-100 dark:border-gray-600 text-gray-400 dark:text-gray-500 cursor-not-allowed" 
                : "border-gray-200 dark:border-gray-600 hover:border-[#4A90E2] dark:hover:border-[#4A90E2] text-[#2C3E50] dark:text-white group-hover:shadow-md hover:text-[#4A90E2] dark:hover:text-blue-400"
            }`}
          >
            {isTranscribing ? (
              <Loader2 className="w-4 h-4 text-[#4A90E2] animate-spin" />
            ) : (
              <Cpu className={`w-4 h-4 ${!audioFile ? 'text-gray-400' : 'text-[#4A90E2]'}`} />
            )}
            Auto-transcribe
          </button>
        </div>
      </div>
      
      <div className="mt-4 flex items-center justify-between">
         <div className="flex items-center gap-1.5 text-xs font-['Inter'] text-[#7F8C8D] dark:text-gray-400">
            <FileText className="w-3.5 h-3.5" />
            <span>{wordCount} words • {charCount} chars</span>
         </div>
         <button 
           onClick={() => setText("")}
           className="text-xs font-['Inter'] font-medium text-[#4A90E2] hover:underline"
         >
           Clear
         </button>
      </div>
    </div>
  );
}
