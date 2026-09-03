import { ReactNode, useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  Activity, AlertTriangle, Database, FileBarChart,
  LayoutDashboard, Menu, Play, RotateCcw, ScrollText, Settings2, Shield, Swords, Workflow, X,
} from "lucide-react";
import { useStore, PRESETS } from "../../store/useStore";
import { Badge, Button, Dot } from "../ui";
import { RunDriver } from "./RunDriver";
import { cx } from "../../lib/format";

const NAV = [
  { group: "Operate", items: [
    { to: "/app", end: true, icon: LayoutDashboard, label: "Overview" },
    { to: "/app/arena", icon: Swords, label: "Arena", hint: "2-player Red vs Blue" },
    { to: "/app/simulation", icon: Activity, label: "Simulation" },
    { to: "/app/workflow", icon: Workflow, label: "Workflow" },
  ]},
  { group: "Investigate", items: [
    { to: "/app/analysis", icon: FileBarChart, label: "Analysis" },
    { to: "/app/evidence", icon: Database, label: "Evidence" },
    { to: "/app/results", icon: Shield, label: "Results" },
  ]},
  { group: "System", items: [
    { to: "/app/logs", icon: ScrollText, label: "Logs" },
    { to: "/app/settings", icon: Settings2, label: "Settings" },
  ]},
];

export function Shell({ children }: { children: ReactNode }) {
  const { mode, liveProbe, banner, dismissBanner, setMode, reset, episode, scenario,
          runState, presetId, loadPreset, logs } = useStore();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname]);

  const warnings = logs.filter((l) => l.level === "warn" || l.level === "error").length;

  return (
    <div className="h-full flex bg-ink-950">
      <RunDriver />
      {/* Sidebar */}
      <aside className={cx(
        "fixed lg:static inset-y-0 left-0 z-40 w-[248px] shrink-0 bg-ink-900 border-r hairline",
        "flex flex-col transition-transform duration-200",
        open ? "translate-x-0" : "-translate-x-full lg:translate-x-0")}>
        <div className="h-14 px-4 flex items-center gap-2.5 border-b hairline">
          <div className="h-7 w-7 rounded-md bg-blue/15 border border-blue/30 grid place-items-center">
            <Swords className="h-3.5 w-3.5 text-blue" />
          </div>
          <div className="leading-none">
            <div className="text-sm font-semibold tracking-tight">ARENA</div>
            <div className="text-2xs text-ink-400 mt-0.5">Tool-Use Security Console</div>
          </div>
          <button className="ml-auto lg:hidden text-ink-400" onClick={() => setOpen(false)}><X className="h-4 w-4" /></button>
        </div>

        <nav className="flex-1 overflow-y-auto px-2.5 py-3 space-y-4">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="label px-2 mb-1.5">{g.group}</div>
              <div className="space-y-0.5">
                {g.items.map((i) => (
                  <NavLink key={i.to} to={i.to} end={(i as any).end}
                    className={({ isActive }) => cx(
                      "group relative flex items-center gap-2.5 rounded-md px-2 py-1.5 text-[13px] transition-colors",
                      isActive ? "bg-ink-800 text-ink-100 font-medium" : "text-ink-300 hover:text-ink-100 hover:bg-ink-850")}>
                    {({ isActive }: any) => (<>
                      {isActive && <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-blue" />}
                      <i.icon className={cx("h-4 w-4 shrink-0", isActive ? "text-blue" : "text-ink-400 group-hover:text-ink-200")} />
                      <span className="truncate">{i.label}</span>
                      {i.label === "Logs" && warnings > 0 && (
                        <span className="ml-auto text-2xs tnum text-amber">{warnings}</span>
                      )}
                    </>)}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* Presenter controls — always reachable, never more than one click away */}
        <div className="border-t hairline p-2.5 space-y-2">
          <div className="label px-1">Presenter</div>
          <select value={presetId} onChange={(e) => loadPreset(e.target.value)}
            className="w-full bg-ink-850 border border-ink-700/70 rounded-md text-xs px-2 h-8 focus-ring text-ink-200">
            {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <div className="flex gap-1.5">
            <Button size="sm" variant="primary" className="flex-1"
              onClick={() => { useStore.getState().loadPreset(presetId); useStore.getState().run(); }}>
              <Play className="h-3 w-3" /> Run Demo
            </Button>
            <Button size="sm" variant="secondary" onClick={reset} title="Reset to a clean presentation state">
              <RotateCcw className="h-3 w-3" />
            </Button>
          </div>
          <button onClick={() => setMode(mode === "demo" ? "live" : "demo")}
            className="w-full flex items-center gap-2 rounded-md bg-ink-850 border border-ink-700/70 px-2 h-8 text-2xs hover:bg-ink-800 transition-colors focus-ring">
            <Dot tone={mode === "demo" ? "amber" : liveProbe === "ok" ? "green" : "red"} pulse={mode === "live"} />
            <span className="text-ink-300">Connection</span>
            <span className="ml-auto font-mono font-medium text-ink-100">{mode === "demo" ? "DEMO" : "LIVE"}</span>
          </button>
        </div>
      </aside>

      {open && <div className="fixed inset-0 z-30 bg-ink-950/60 lg:hidden" onClick={() => setOpen(false)} />}

      {/* Main */}
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-14 shrink-0 border-b hairline bg-ink-900/80 backdrop-blur px-4 flex items-center gap-3">
          <button className="lg:hidden text-ink-300" onClick={() => setOpen(true)} aria-label="Open navigation"><Menu className="h-5 w-5" /></button>
          <div className="min-w-0 flex items-center gap-2.5">
            <span className="text-sm font-medium truncate">{scenario?.id ?? "No episode"}</span>
            {scenario && (
              <>
                <Badge tone="neutral" mono>{scenario.domain}</Badge>
                {scenario.adversarial
                  ? <Badge tone="red">adversarial · {scenario.objective!.family.replace(/_/g, " ")}</Badge>
                  : <Badge tone="green">benign</Badge>}
              </>
            )}
          </div>
          <div className="ml-auto flex items-center gap-2.5">
            {episode && (
              <div className="hidden md:flex items-center gap-2 text-2xs text-ink-400">
                <span className="tnum">step {episode.stepIndex}/{episode.scenario.maxSteps}</span>
                <span className="h-3 w-px bg-ink-700" />
                <span className="tnum">{episode.history.length} calls</span>
              </div>
            )}
            <Badge tone={runState === "running" ? "blue" : runState === "done" ? "green" : "neutral"}>
              <Dot tone={runState === "running" ? "blue" : runState === "done" ? "green" : "neutral"} pulse={runState === "running"} />
              {runState}
            </Badge>
          </div>
        </header>

        {banner && (
          <div className="shrink-0 flex items-center gap-2.5 px-4 py-2 bg-amber-wash border-b border-amber/30 text-xs text-amber">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
            <span className="flex-1">{banner}</span>
            <button onClick={dismissBanner} className="text-amber/70 hover:text-amber"><X className="h-3.5 w-3.5" /></button>
          </div>
        )}

        <main className="flex-1 min-h-0 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}


