import { FileText, Cpu } from "lucide-react";

export function TextInputCard() {
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

      <div className="relative flex-1">
        <textarea
          className="w-full h-full min-h-[160px] p-4 sm:p-5 bg-[#F5F7FA] dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 focus:border-[#4A90E2] dark:focus:border-[#4A90E2] focus:ring-1 focus:ring-[#4A90E2] outline-none resize-none font-['Inter'] text-sm sm:text-base text-[#2C3E50] dark:text-gray-200 leading-relaxed transition-colors group-hover:bg-white dark:group-hover:bg-gray-800"
          placeholder="Enter transcript manually or auto-transcribe from audio..."
          defaultValue={"I'm honestly thrilled with the recent update to the platform. It's so much faster and exactly what we needed for our team. Thank you for listening to our feedback!"}
        ></textarea>
        
        <div className="absolute bottom-4 right-4">
          <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 hover:border-[#4A90E2] dark:hover:border-[#4A90E2] shadow-sm rounded-lg text-[#2C3E50] dark:text-white font-['Inter'] text-sm font-medium transition-all group-hover:shadow-md hover:text-[#4A90E2] dark:hover:text-blue-400">
            <Cpu className="w-4 h-4 text-[#4A90E2]" />
            Auto-transcribe
          </button>
        </div>
      </div>
      
      <div className="mt-4 flex items-center justify-between">
         <div className="flex items-center gap-1.5 text-xs font-['Inter'] text-[#7F8C8D] dark:text-gray-400">
            <FileText className="w-3.5 h-3.5" />
            <span>31 words • 182 chars</span>
         </div>
         <button className="text-xs font-['Inter'] font-medium text-[#4A90E2] hover:underline">Clear</button>
      </div>
    </div>
  );
}
