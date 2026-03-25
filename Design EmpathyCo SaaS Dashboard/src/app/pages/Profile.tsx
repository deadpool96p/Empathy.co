import { User, Mail, Settings, Bell, Palette, ChevronRight, LogOut, CheckCircle, Moon, Sun } from "lucide-react";
import { useForm } from "react-hook-form";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";

type ProfileFormData = {
  fullName: string;
  email: string;
};

export function Profile() {
  const [isSaved, setIsSaved] = useState(false);
  const [showAppearance, setShowAppearance] = useState(false);
  const [isDark, setIsDark] = useState(false);
  
  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggleDarkMode = () => {
    const nextDark = !isDark;
    setIsDark(nextDark);
    if (nextDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const { register, handleSubmit } = useForm<ProfileFormData>({
    defaultValues: {
      fullName: "Jane Doe",
      email: "jane.doe@example.com",
    },
  });

  const onSubmit = (data: ProfileFormData) => {
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 2500);
  };

  return (
    <div className="flex flex-col gap-6 w-full max-w-lg mx-auto pb-6">
      <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 sm:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:shadow-none border border-gray-50 dark:border-gray-800 flex items-center gap-5">
        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-100 to-purple-100 border-4 border-white dark:border-gray-800 shadow-md flex flex-col items-center justify-center relative overflow-hidden">
          <span className="font-['Poppins'] text-2xl font-bold text-[#4A90E2]">JD</span>
          <div className="absolute inset-x-0 bottom-0 bg-black/20 dark:bg-black/40 text-white text-[10px] font-medium font-['Inter'] text-center py-0.5 cursor-pointer hover:bg-black/30 dark:hover:bg-black/60 transition-colors">Edit</div>
        </div>
        <div>
          <h2 className="font-['Poppins'] font-semibold text-xl text-[#2C3E50] dark:text-white">Jane Doe</h2>
          <span className="inline-block mt-1 font-['Inter'] text-xs font-medium px-2.5 py-1 bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-full">Pro Member</span>
        </div>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="bg-white dark:bg-gray-900 rounded-2xl p-6 sm:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:shadow-none border border-gray-50 dark:border-gray-800 flex flex-col gap-5">
        <h3 className="font-['Poppins'] font-semibold text-lg text-[#2C3E50] dark:text-white mb-2 flex items-center gap-2">
          <Settings className="w-5 h-5 text-[#4A90E2]" />
          Account Settings
        </h3>

        <div className="flex flex-col gap-2">
          <label className="font-['Inter'] text-sm font-medium text-[#7F8C8D] dark:text-gray-400 flex items-center gap-1.5">
            <User className="w-4 h-4" /> Full Name
          </label>
          <input
            {...register("fullName", { required: true })}
            className="w-full px-4 py-3 bg-[#F5F7FA] dark:bg-gray-800 border border-gray-200 dark:border-gray-700 focus:border-[#4A90E2] dark:focus:border-[#4A90E2] focus:ring-1 focus:ring-[#4A90E2] outline-none rounded-xl font-['Inter'] text-[#2C3E50] dark:text-white transition-colors"
          />
        </div>

        <div className="flex flex-col gap-2">
          <label className="font-['Inter'] text-sm font-medium text-[#7F8C8D] dark:text-gray-400 flex items-center gap-1.5">
            <Mail className="w-4 h-4" /> Email Address
          </label>
          <input
            type="email"
            {...register("email", { required: true })}
            className="w-full px-4 py-3 bg-[#F5F7FA] dark:bg-gray-800 border border-gray-200 dark:border-gray-700 focus:border-[#4A90E2] dark:focus:border-[#4A90E2] focus:ring-1 focus:ring-[#4A90E2] outline-none rounded-xl font-['Inter'] text-[#2C3E50] dark:text-white transition-colors"
          />
        </div>

        <motion.button
          whileTap={{ scale: 0.98 }}
          type="submit"
          className="mt-2 w-full py-3.5 bg-[#4A90E2] hover:bg-[#357ABD] active:bg-[#2C6EAF] text-white font-['Poppins'] font-semibold rounded-xl shadow-[0_4px_15px_rgba(74,144,226,0.3)] transition-all flex items-center justify-center gap-2"
        >
          {isSaved ? <><CheckCircle className="w-5 h-5" /> Saved!</> : "Save Changes"}
        </motion.button>
      </form>

      <div className="bg-white dark:bg-gray-900 rounded-2xl p-2 shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:shadow-none border border-gray-50 dark:border-gray-800">
        <button className="w-full flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-xl transition-colors text-left group">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-orange-50 dark:bg-orange-900/20 flex items-center justify-center">
              <Bell className="w-4 h-4 text-orange-500" />
            </div>
            <span className="font-['Inter'] font-medium text-[#2C3E50] dark:text-white">Notifications</span>
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-[#4A90E2]" />
        </button>

        <div className="rounded-xl overflow-hidden transition-colors bg-white dark:bg-gray-900">
          <button 
            onClick={() => setShowAppearance(!showAppearance)}
            className="w-full flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors text-left group"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-purple-50 dark:bg-purple-900/20 flex items-center justify-center">
                <Palette className="w-4 h-4 text-purple-500" />
              </div>
              <span className="font-['Inter'] font-medium text-[#2C3E50] dark:text-white">Appearance</span>
            </div>
            <ChevronRight className={`w-4 h-4 text-gray-400 group-hover:text-[#4A90E2] transition-transform ${showAppearance ? 'rotate-90' : ''}`} />
          </button>
          
          <AnimatePresence>
            {showAppearance && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-4 pt-2 ml-11">
                  <div className="flex items-center justify-between bg-gray-50 dark:bg-gray-800 p-3 rounded-xl border border-gray-100 dark:border-gray-700">
                    <div className="flex items-center gap-2 text-sm font-['Inter'] font-medium text-[#2C3E50] dark:text-white">
                      {isDark ? <Moon className="w-4 h-4 text-blue-400" /> : <Sun className="w-4 h-4 text-yellow-500" />}
                      Dark Theme
                    </div>
                    <button
                      onClick={toggleDarkMode}
                      className={`w-11 h-6 rounded-full transition-colors flex items-center px-1 ${
                        isDark ? 'bg-[#4A90E2]' : 'bg-gray-300'
                      }`}
                    >
                      <motion.div
                        layout
                        className="w-4 h-4 rounded-full bg-white shadow-sm"
                        animate={{ x: isDark ? 20 : 0 }}
                        transition={{ type: "spring", stiffness: 500, damping: 30 }}
                      />
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <hr className="my-1 border-gray-100 dark:border-gray-800 mx-4" />
        <button className="w-full flex items-center gap-3 p-4 hover:bg-red-50 dark:hover:bg-red-900/10 rounded-xl transition-colors text-left text-[#FF6B6B] font-['Inter'] font-medium">
          <div className="w-8 h-8 rounded-lg bg-red-100/50 dark:bg-red-900/20 flex items-center justify-center">
            <LogOut className="w-4 h-4" />
          </div>
          Sign Out
        </button>
      </div>
    </div>
  );
}
