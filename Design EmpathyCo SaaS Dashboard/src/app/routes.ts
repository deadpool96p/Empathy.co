import { createBrowserRouter } from "react-router";
import { MobileLayout } from "./layouts/MobileLayout";
import { Dashboard } from "./pages/Dashboard";
import { Profile } from "./pages/Profile";
import { HistoryPage } from "./pages/HistoryPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: MobileLayout,
    children: [
      { index: true, Component: Dashboard },
      { path: "profile", Component: Profile },
      { path: "history", Component: HistoryPage },
    ],
  },
]);
