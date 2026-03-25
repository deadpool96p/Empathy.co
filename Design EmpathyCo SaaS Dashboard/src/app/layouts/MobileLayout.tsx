import { Outlet, useLocation } from "react-router";
import { MobileHeader } from "../components/MobileHeader";
import { BottomNav } from "../components/BottomNav";
import { motion, AnimatePresence } from "motion/react";

export function MobileLayout() {
  const location = useLocation();

  return (
    <div className="flex flex-col h-screen h-[100dvh] bg-[#F5F7FA] dark:bg-gray-950 overflow-hidden font-['Inter'] transition-colors">
      <MobileHeader />
      <main className="flex-1 overflow-y-auto px-4 pb-28 pt-4 custom-scrollbar">
        <div className="max-w-[800px] mx-auto w-full relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
      <BottomNav />
    </div>
  );
}
