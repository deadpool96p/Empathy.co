import { Bot, Bell, X } from "lucide-react";
import { Link } from "react-router";
import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";

export function MobileHeader() {
  const [showNotifications, setShowNotifications] = useState(false);

  return (
    <header className="w-full bg-white/80 dark:bg-gray-900/80 backdrop-blur-md sticky top-0 z-40 border-b border-gray-100 dark:border-gray-800 px-4 py-3 flex items-center justify-between shadow-sm">
      <Link to="/" className="flex items-center gap-2.5">
        <div className="w-8 h-8 bg-gradient-to-br from-[#4A90E2] to-[#357ABD] rounded-lg flex items-center justify-center shadow-md shadow-blue-500/20">
          <Bot className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="font-['Poppins'] font-semibold text-lg text-[#2C3E50] dark:text-white leading-none">
            EmpathyCo
          </h1>
        </div>
      </Link>
      
      <div className="flex items-center gap-3">
        <div className="relative">
          <button 
            onClick={() => setShowNotifications(!showNotifications)}
            className="relative w-9 h-9 flex items-center justify-center text-gray-400 hover:text-[#4A90E2] dark:hover:text-[#4A90E2] transition-colors rounded-full hover:bg-blue-50 dark:hover:bg-gray-800"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute top-2 right-2.5 w-2 h-2 bg-[#FF6B6B] rounded-full border border-white dark:border-gray-900"></span>
          </button>

          <AnimatePresence>
            {showNotifications && (
              <>
                <div 
                  className="fixed inset-0 z-40"
                  onClick={() => setShowNotifications(false)}
                />
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="absolute right-0 mt-2 w-72 bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-100 dark:border-gray-700 z-50 overflow-hidden"
                >
                  <div className="flex items-center justify-between p-4 border-b border-gray-50 dark:border-gray-700/50">
                    <h3 className="font-['Poppins'] font-semibold text-[#2C3E50] dark:text-white text-sm">Notifications</h3>
                    <button onClick={() => setShowNotifications(false)} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="max-h-[300px] overflow-y-auto">
                    {[
                      { title: "Analysis Complete", desc: "Customer Call 142 finished.", time: "2m ago", unread: true },
                      { title: "Weekly Report", desc: "Your weekly summary is ready.", time: "1h ago", unread: false },
                      { title: "System Update", desc: "New emotion models available.", time: "1d ago", unread: false }
                    ].map((notif, i) => (
                      <div key={i} className={`p-4 border-b border-gray-50 dark:border-gray-700/50 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors cursor-pointer ${notif.unread ? 'bg-blue-50/50 dark:bg-blue-900/10' : ''}`}>
                        <div className="flex justify-between items-start mb-1">
                          <h4 className={`text-sm font-semibold ${notif.unread ? 'text-[#2C3E50] dark:text-white' : 'text-gray-600 dark:text-gray-300'}`}>{notif.title}</h4>
                          <span className="text-[10px] text-gray-400">{notif.time}</span>
                        </div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">{notif.desc}</p>
                      </div>
                    ))}
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-800/50 text-center border-t border-gray-100 dark:border-gray-700">
                    <button className="text-xs font-semibold text-[#4A90E2] hover:text-[#357ABD] transition-colors">Mark all as read</button>
                  </div>
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>
        <Link to="/profile">
          <div className="w-8 h-8 rounded-full bg-gray-200 border-2 border-white dark:border-gray-800 shadow-sm overflow-hidden flex items-center justify-center active:scale-95 transition-transform">
            <span className="font-['Inter'] text-xs font-medium text-gray-500 dark:text-gray-700">JD</span>
          </div>
        </Link>
      </div>
    </header>
  );
}
