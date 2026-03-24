import { useState, useEffect } from "react";
import { AudioInput } from "./components/AudioInput";
import { TextInput } from "./components/TextInput";
import { ActionBar } from "./components/ActionBar";
import { ResultsPanel } from "./components/ResultsPanel";
import { LoadingOverlay } from "./components/LoadingOverlay";
import type { AnalysisResponse } from "./services/types";
import { analyze, getModels } from "./services/api";
import { toast } from "react-toastify";
import { BrainCircuit } from "lucide-react";

function App() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("multi");
  const [models, setModels] = useState<string[]>(["en", "hi", "mr", "multi"]);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResponse | null>(null);

  useEffect(() => {
    // Fetch available models on load
    const fetchModels = async () => {
      try {
        const res = await getModels();
        if (res.models && res.models.length > 0) {
          setModels(res.models);
          if (!res.models.includes("multi") && res.models.length > 0) {
             setLanguage(res.models[0]);
          }
        }
      } catch (err) {
        toast.error("Could not fetch models. Using defaults.");
        console.error("Failed to load models list", err);
      }
    };
    fetchModels();
  }, []);

  // Clear results if input changes to avoid stale data
  useEffect(() => {
    if (results) setResults(null);
  }, [audioFile, text, language]);

  const handleAnalyze = async () => {
    if (!audioFile && !text.trim()) {
      toast.warning("Please provide either audio or text input.");
      return;
    }

    setIsAnalyzing(true);
    setResults(null);
    try {
      const response = await analyze(audioFile, text, language);
      setResults(response);
      toast.success("Analysis complete!");
    } catch (err: any) {
      toast.error(err.message || "Cannot connect to server. Ensure backend is running.");
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClearAll = () => {
    setAudioFile(null);
    setText("");
    setResults(null);
  };

  const canAnalyze = !!audioFile || text.trim().length > 0;

  return (
    <div className="min-h-screen bg-[#F5F7FA] dark:bg-[#111827] text-gray-900 dark:text-gray-100 font-['Inter'] selection:bg-blue-100 selection:text-blue-900">
      <LoadingOverlay isVisible={isAnalyzing} message="Analyzing input..." />
      
      {/* Header */}
      <header className="sticky top-0 z-40 w-full backdrop-blur-md bg-white/70 dark:bg-gray-900/70 border-b border-gray-200/50 dark:border-gray-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
             <div className="w-10 h-10 bg-gradient-to-br from-[#4A90E2] to-[#357ABD] rounded-xl shadow-lg shadow-blue-500/20 flex items-center justify-center">
               <BrainCircuit className="w-6 h-6 text-white" />
             </div>
             <h1 className="font-['Poppins'] font-bold text-xl tracking-tight text-[#2C3E50] dark:text-white">
               Empathy<span className="text-[#4A90E2]">Co</span>
             </h1>
          </div>
          <button 
             onClick={handleClearAll}
             className="text-sm font-medium text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
          >
             Clear Session
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <div className="max-w-4xl mx-auto flex flex-col gap-6 sm:gap-8">
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8">
            <AudioInput 
              selectedFile={audioFile} 
              onAudioSelected={setAudioFile} 
            />
            <TextInput 
              text={text} 
              setText={setText} 
              audioFile={audioFile} 
            />
          </div>

          <ActionBar 
            language={language}
            setLanguage={setLanguage}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            canAnalyze={canAnalyze}
            models={models}
          />

          {/* Conditional Rendering of Results or Placeholder */}
          {results ? (
            <div className="mt-4">
              <ResultsPanel results={results} />
            </div>
          ) : (
             <div className="mt-8 flex flex-col items-center justify-center text-center p-8 sm:p-12 border-2 border-dashed border-gray-200 dark:border-gray-800 rounded-3xl opacity-50 transition-opacity">
               <BrainCircuit className="w-16 h-16 text-gray-300 dark:text-gray-700 mb-4" />
               <p className="font-['Inter'] font-medium text-gray-500 dark:text-gray-400">
                 Provide input and click Analyze to view emotional insights
               </p>
             </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
