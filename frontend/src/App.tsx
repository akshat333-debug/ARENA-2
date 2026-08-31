import { useEffect } from "react";
import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./components/layout/Shell";
import { useStore } from "./store/useStore";
import Home from "./pages/Home";
import Overview from "./pages/dashboard/Overview";
import Arena from "./pages/dashboard/Arena";
import Simulation from "./pages/dashboard/Simulation";
import Workflow from "./pages/dashboard/Workflow";
import Analysis from "./pages/dashboard/Analysis";
import Evidence from "./pages/dashboard/Evidence";
import Results from "./pages/dashboard/Results";
import Logs from "./pages/dashboard/Logs";
import Settings from "./pages/dashboard/Settings";

/** Boot with a scenario already loaded. An empty dashboard is a bad first
 *  second of a demo — the presenter should be able to interact immediately. */
function Boot({ children }: { children: React.ReactNode }) {
  const { episode, loadPreset } = useStore();
  useEffect(() => { if (!episode) loadPreset("std"); }, []);
  return <>{children}</>;
}

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/app/*" element={
          <Boot>
            <Shell>
              <Routes>
                <Route index element={<Overview />} />
                <Route path="arena" element={<Arena />} />
                <Route path="simulation" element={<Simulation />} />
                <Route path="workflow" element={<Workflow />} />
                <Route path="analysis" element={<Analysis />} />
                <Route path="evidence" element={<Evidence />} />
                <Route path="results" element={<Results />} />
                <Route path="logs" element={<Logs />} />
                <Route path="settings" element={<Settings />} />
                <Route path="*" element={<Navigate to="/app" replace />} />
              </Routes>
            </Shell>
          </Boot>
        } />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HashRouter>
  );
}
