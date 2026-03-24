import { useRef, useState, useEffect } from "react";
import { Upload, Mic, Play, Square, CheckCircle, Trash2 } from "lucide-react";

interface AudioInputProps {
  onAudioSelected: (file: File | null) => void;
  selectedFile: File | null;
}

export function AudioInput({ onAudioSelected, selectedFile }: AudioInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    // Cleanup audio resources
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onAudioSelected(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.includes('audio')) {
        onAudioSelected(file);
      }
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const file = new File([audioBlob], "recorded_audio.wav", { type: 'audio/wav' });
        onAudioSelected(file);
        
        // Stop all tracks to turn off microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      timerRef.current = window.setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 4) { // 5 seconds max (0 to 4 is 5 ticks)
            stopRecording();
            return 5;
          }
          return prev + 1;
        });
      }, 1000);
      
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Microphone access denied or not available.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const handlePlayPause = () => {
    if (!selectedFile) return;

    if (!audioRef.current) {
      const url = URL.createObjectURL(selectedFile);
      audioRef.current = new Audio(url);
      audioRef.current.onended = () => setIsPlaying(false);
    }

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  // Re-create audio ref when file changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
      audioRef.current = null;
    }
  }, [selectedFile]);

  // Format bytes
  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

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

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".wav,.mp3,.flac,.m4a,audio/*"
        className="hidden"
      />

      <div 
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className="border-2 border-dashed border-gray-200 dark:border-gray-700 hover:border-[#4A90E2] dark:hover:border-[#4A90E2] transition-colors rounded-xl p-6 sm:p-8 flex flex-col items-center justify-center flex-1 bg-gray-50/50 dark:bg-gray-800/50 cursor-pointer group min-h-[160px]"
      >
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
        {isRecording ? (
          <button 
            onClick={stopRecording}
            className="flex items-center justify-center gap-2 flex-1 py-2.5 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-xl font-['Inter'] text-sm font-medium transition-colors border border-red-100 dark:border-red-800 animate-pulse"
          >
            <Square className="w-4 h-4" fill="currentColor" />
            Stop ({5 - recordingTime}s)
          </button>
        ) : (
          <button 
            onClick={startRecording}
            className="flex items-center justify-center gap-2 flex-1 py-2.5 bg-[#F5F7FA] dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-[#2C3E50] dark:text-white rounded-xl font-['Inter'] text-sm font-medium transition-colors border border-gray-100 dark:border-gray-700"
          >
            <Mic className="w-4 h-4 text-[#FF6B6B]" />
            Record 5s
          </button>
        )}
        
        <button 
          onClick={handlePlayPause}
          disabled={!selectedFile}
          className={`w-10 h-10 flex items-center justify-center rounded-xl transition-colors border ${
            selectedFile 
              ? "bg-[#F5F7FA] dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-[#2C3E50] dark:text-white border-gray-100 dark:border-gray-700 cursor-pointer" 
              : "bg-gray-50 dark:bg-gray-800/30 text-gray-300 dark:text-gray-600 border-gray-50 dark:border-gray-800/30 cursor-not-allowed"
          }`}
        >
          {isPlaying ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" fill={selectedFile ? "currentColor" : "none"}/>}
        </button>
      </div>

      {/* Visualizer & File info */}
      {selectedFile && (
        <div className="mt-5 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-100 dark:border-gray-700 flex items-center gap-4">
          <div className="flex-1 flex items-center gap-1 sm:gap-1.5 h-8 overflow-hidden">
            {[...Array(30)].map((_, i) => (
              <div
                key={i}
                className="w-1.5 bg-[#4A90E2] rounded-full transition-all duration-300"
                style={{
                  height: isPlaying ? `${Math.max(20, Math.random() * 100)}%` : '20%',
                  opacity: i % 2 === 0 ? 0.8 : 0.4,
                }}
              ></div>
            ))}
          </div>
          <div className="flex flex-col items-end flex-shrink-0">
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-xs text-[#2C3E50] dark:text-gray-200 font-medium truncate max-w-[120px]" title={selectedFile.name}>
                {selectedFile.name}
              </span>
              <CheckCircle className="w-3.5 h-3.5 text-[#2ECC71]" />
            </div>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="font-['Inter'] text-[10px] text-[#7F8C8D] dark:text-gray-400">{formatBytes(selectedFile.size)}</span>
              <button onClick={() => onAudioSelected(null)} className="text-[#FF6B6B] hover:text-red-700 dark:hover:text-red-400">
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
