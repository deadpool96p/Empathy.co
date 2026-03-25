import { Bot } from "lucide-react";

export function Header() {
  return (
    <header className="w-full bg-transparent py-8 px-10 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-gradient-to-br from-[#4A90E2] to-[#357ABD] rounded-xl flex items-center justify-center shadow-md shadow-blue-500/20">
          <Bot className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="font-['Poppins'] font-semibold text-2xl text-[#2C3E50] leading-none">
            EmpathyCo
          </h1>
          <p className="font-['Inter'] text-sm text-[#7F8C8D] mt-1">
            Multimodal Emotion Intelligence
          </p>
        </div>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3 px-4 py-2 bg-white rounded-full shadow-[0_4px_20px_-4px_rgba(0,0,0,0.05)]">
          <div className="w-2 h-2 rounded-full bg-[#2ECC71] animate-pulse"></div>
          <span className="font-['Inter'] text-sm font-medium text-[#2C3E50]">API Online</span>
        </div>
        <div className="w-10 h-10 rounded-full bg-gray-200 border-2 border-white shadow-sm overflow-hidden flex items-center justify-center">
          <span className="font-['Inter'] text-sm font-medium text-gray-500">JD</span>
        </div>
      </div>
    </header>
  );
}
