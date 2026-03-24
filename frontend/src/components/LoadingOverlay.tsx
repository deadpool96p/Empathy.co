import { Loader2 } from "lucide-react";

interface LoadingOverlayProps {
  isVisible: boolean;
  message?: string;
}

export function LoadingOverlay({ isVisible, message = "Analyzing..." }: LoadingOverlayProps) {
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white/60 dark:bg-gray-900/80 backdrop-blur-sm transition-all duration-300 animate-in fade-in">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-2xl border border-gray-100 dark:border-gray-700 flex flex-col items-center gap-4">
        <Loader2 className="w-10 h-10 text-[#4A90E2] animate-spin" />
        <p className="font-['Inter'] font-medium text-[#2C3E50] dark:text-gray-200 text-sm">
          {message}
        </p>
      </div>
    </div>
  );
}
