import { useState } from "react";
import { History, ChevronLeft, ChevronRight, FileAudio, FileText } from "lucide-react";

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  const historyItems = [
    { id: 1, title: "Customer Call 142", date: "Today, 10:45 AM", type: "audio" },
    { id: 2, title: "Support Ticket 89", date: "Today, 09:12 AM", type: "text" },
    { id: 3, title: "Sales Demo Q3", date: "Yesterday, 14:30", type: "audio" },
    { id: 4, title: "Feedback Form", date: "Yesterday, 11:20", type: "text" },
    { id: 5, title: "User Interview", date: "Mar 20, 16:00", type: "audio" },
  ];

  return (
    <aside
      className={`h-screen bg-white border-r border-gray-100 flex flex-col transition-all duration-300 relative shadow-[4px_0_24px_-4px_rgba(0,0,0,0.02)] ${
        collapsed ? "w-20" : "w-64"
      }`}
    >
      <div className="flex items-center justify-between p-6 border-b border-gray-100">
        {!collapsed && (
          <h2 className="font-['Poppins'] font-semibold text-[#2C3E50] text-lg flex items-center gap-2">
            <History className="w-5 h-5 text-[#4A90E2]" />
            History
          </h2>
        )}
        {collapsed && <History className="w-6 h-6 text-[#4A90E2] mx-auto" />}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className={`text-[#7F8C8D] hover:bg-gray-50 p-1.5 rounded-lg transition-colors ${
            collapsed ? "absolute -right-3 top-6 bg-white border border-gray-100 shadow-sm" : ""
          }`}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-5 h-5" />}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {historyItems.map((item) => (
          <button
            key={item.id}
            className={`w-full flex items-center gap-3 p-3 rounded-xl hover:bg-[#F5F7FA] transition-colors group text-left ${
              collapsed ? "justify-center" : "justify-start"
            }`}
          >
            <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center flex-shrink-0">
              {item.type === "audio" ? (
                <FileAudio className="w-4 h-4 text-[#4A90E2]" />
              ) : (
                <FileText className="w-4 h-4 text-[#4A90E2]" />
              )}
            </div>
            {!collapsed && (
              <div className="flex-1 min-w-0">
                <p className="font-['Inter'] text-sm font-medium text-[#2C3E50] truncate group-hover:text-[#4A90E2] transition-colors">
                  {item.title}
                </p>
                <p className="font-['Inter'] text-xs text-[#7F8C8D] truncate">
                  {item.date}
                </p>
              </div>
            )}
          </button>
        ))}
      </div>
    </aside>
  );
}
