import { FileAudio, FileText, Search, MoreVertical } from "lucide-react";

export function HistoryPage() {
  const historyItems = [
    { id: 1, title: "Customer Call 142", date: "Today, 10:45 AM", type: "audio", emotion: "Happy", score: "92%" },
    { id: 2, title: "Support Ticket 89", date: "Today, 09:12 AM", type: "text", emotion: "Frustrated", score: "78%" },
    { id: 3, title: "Sales Demo Q3", date: "Yesterday, 14:30", type: "audio", emotion: "Excited", score: "85%" },
    { id: 4, title: "Feedback Form", date: "Yesterday, 11:20", type: "text", emotion: "Neutral", score: "60%" },
    { id: 5, title: "User Interview", date: "Mar 20, 16:00", type: "audio", emotion: "Calm", score: "88%" },
    { id: 6, title: "Q1 Townhall", date: "Mar 15, 09:00", type: "audio", emotion: "Joy", score: "94%" },
    { id: 7, title: "Complaint Email", date: "Mar 12, 13:45", type: "text", emotion: "Angry", score: "91%" },
  ];

  return (
    <div className="flex flex-col gap-6 w-full max-w-2xl mx-auto pb-6">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-['Poppins'] font-bold text-2xl text-[#2C3E50] dark:text-white">
          Analysis History
        </h2>
        <span className="text-xs font-['Inter'] font-medium px-3 py-1 bg-blue-50 dark:bg-blue-900/30 text-[#4A90E2] dark:text-blue-400 rounded-full">
          {historyItems.length} items
        </span>
      </div>

      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
          <Search className="w-5 h-5 text-gray-400" />
        </div>
        <input
          type="text"
          placeholder="Search by title, emotion..."
          className="w-full pl-11 pr-4 py-3.5 bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 shadow-[0_4px_20px_rgb(0,0,0,0.03)] dark:shadow-none focus:border-[#4A90E2] dark:focus:border-[#4A90E2] focus:ring-1 focus:ring-[#4A90E2] outline-none rounded-xl font-['Inter'] text-[#2C3E50] dark:text-white transition-colors"
        />
      </div>

      <div className="space-y-3">
        {historyItems.map((item) => (
          <div
            key={item.id}
            className="group flex flex-col sm:flex-row sm:items-center justify-between p-4 bg-white dark:bg-gray-900 rounded-xl shadow-[0_4px_15px_rgb(0,0,0,0.02)] dark:shadow-none border border-gray-50 dark:border-gray-800 hover:border-blue-100 dark:hover:border-gray-700 hover:shadow-md transition-all gap-4 cursor-pointer"
          >
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-blue-50 dark:bg-blue-900/20 flex items-center justify-center flex-shrink-0">
                {item.type === "audio" ? (
                  <FileAudio className="w-5 h-5 text-[#4A90E2]" />
                ) : (
                  <FileText className="w-5 h-5 text-[#4A90E2]" />
                )}
              </div>
              <div>
                <h3 className="font-['Poppins'] font-semibold text-[#2C3E50] dark:text-gray-200 group-hover:text-[#4A90E2] dark:group-hover:text-blue-400 transition-colors line-clamp-1 text-base">
                  {item.title}
                </h3>
                <p className="font-['Inter'] text-xs text-[#7F8C8D] dark:text-gray-400 mt-0.5 flex items-center gap-1.5">
                  <span>{item.date}</span>
                  <span className="w-1 h-1 bg-gray-300 dark:bg-gray-600 rounded-full"></span>
                  <span className="uppercase text-[10px] tracking-wider font-semibold">{item.type}</span>
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between sm:justify-end gap-4 ml-16 sm:ml-0">
              <div className="flex flex-col items-start sm:items-end">
                <span className={`font-['Poppins'] text-sm font-semibold ${item.emotion === 'Angry' || item.emotion === 'Frustrated' ? 'text-[#FF6B6B]' : item.emotion === 'Happy' || item.emotion === 'Joy' ? 'text-[#2ECC71]' : 'text-[#4A90E2] dark:text-blue-400'}`}>
                  {item.emotion}
                </span>
                <span className="font-['Inter'] text-xs text-[#7F8C8D] dark:text-gray-400">
                  {item.score} Match
                </span>
              </div>
              <button className="p-2 text-gray-400 hover:text-[#4A90E2] dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg transition-colors">
                <MoreVertical className="w-5 h-5" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
