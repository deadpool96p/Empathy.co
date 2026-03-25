import { Home, History, User } from "lucide-react";
import { Link, useLocation } from "react-router";
import { motion } from "motion/react";

export function BottomNav() {
  const location = useLocation();

  const navItems = [
    { name: "Dashboard", path: "/", icon: Home },
    { name: "History", path: "/history", icon: History },
    { name: "Profile", path: "/profile", icon: User },
  ];

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-900 border-t border-gray-100 dark:border-gray-800 px-6 py-3 pb-safe z-50 shadow-[0_-8px_30px_rgba(0,0,0,0.04)] dark:shadow-[0_-8px_30px_rgba(0,0,0,0.4)] transition-colors">
      <div className="flex items-center justify-between max-w-sm mx-auto w-full">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`relative flex flex-col items-center gap-1 p-2 rounded-xl transition-colors ${
                isActive ? "text-[#4A90E2]" : "text-[#7F8C8D] dark:text-gray-400 hover:text-[#2C3E50] dark:hover:text-gray-200"
              }`}
            >
              <div className="relative">
                <Icon className={`w-6 h-6 transition-all duration-300 ${isActive ? "scale-110" : "scale-100"}`} strokeWidth={isActive ? 2.5 : 2} />
                {isActive && (
                  <motion.div
                    layoutId="bottom-nav-indicator"
                    className="absolute -bottom-6 left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-[#4A90E2] rounded-full shadow-[0_0_8px_rgba(74,144,226,0.6)]"
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  />
                )}
              </div>
              <span className={`font-['Inter'] text-[10px] font-semibold transition-all duration-300 ${isActive ? "opacity-100 mt-1" : "opacity-0 h-0"}`}>
                {item.name}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
