import { RotateCcw, TriangleAlert } from "lucide-react";
import { useStore, PRESETS } from "../../store/useStore";
import { Badge, Button, Panel, SegMap } from "../../components/ui";
import { DEFENDERS } from "../../lib/policies";
import { FAMILIES } from "../../lib/domain";
import { num } from "../../lib/format";

export default function Settings() {
  const s = useStore();
  const w = s.weights;
  // The invariant the whole reward design rests on.
  const tie = Math.abs((0.5 * w.wDetect - 0.5 * w.wFalseQuarantine) - (-0.5 * w.wMiss)) < 1e-9;

  return (
    <div className="p-4 lg:p-5 grid lg:grid-cols-2 gap-4 items-start">
      <Panel title="Scenario generator" subtitle="Mirrors ScenarioConfig in arena/config.py. Scale is configuration, never code.">
        <div className="space-y-4">
          <Slider label="Adversarial ratio" value={s.gen.adversarialRatio} min={0} max={1} step={0.05}
            fmt={(v) => `${(v * 100).toFixed(0)}%`}
            hint="Share of sessions that carry an attack objective."
            onChange={(v) => s.setGen({ adversarialRatio: v })} />
          <Slider label="Max steps" value={s.gen.maxSteps} min={4} max={24} step={1}
            fmt={(v) => String(v)} hint="Session budget. Observable to both sides, so the cap is a real terminal."
            onChange={(v) => s.setGen({ maxSteps: v })} />
          <Slider label="Tools per registry (max)" value={s.gen.nToolsMax} min={6} max={14} step={1}
            fmt={(v) => String(v)} hint="Every registry is still guaranteed a source → propagator → sink chain."
            onChange={(v) => s.setGen({ nToolsMax: Math.max(v, s.gen.nToolsMin) })} />
          <div>
            <div className="label mb-2">Attack families in play</div>
            <div className="flex flex-wrap gap-1.5">
              {FAMILIES.map((f) => {
                const on = s.gen.families.includes(f.family);
                return (
                  <button key={f.family} onClick={() => s.setGen({
                    families: on
                      ? (s.gen.families.length > 1 ? s.gen.families.filter((x) => x !== f.family) : s.gen.families)
                      : [...s.gen.families, f.family],
                  })}>
                    <Badge tone={on ? "red" : "neutral"}>{f.label}</Badge>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="Reward weights" subtitle="Mirrors RewardConfig. Changing these changes what 'good defence' means.">
        <div className="space-y-4">
          <Slider label="w_detect" value={w.wDetect} min={0} max={3} step={0.1} fmt={num2}
            hint="Credit for stopping an attack in flight." onChange={(v) => s.setWeights({ wDetect: v })} />
          <Slider label="w_miss" value={w.wMiss} min={0} max={3} step={0.1} fmt={num2}
            hint="Penalty for letting an attack complete." onChange={(v) => s.setWeights({ wMiss: v })} />
          <Slider label="w_false_quarantine" value={w.wFalseQuarantine} min={0} max={4} step={0.1} fmt={num2}
            hint="Penalty for killing a benign session." onChange={(v) => s.setWeights({ wFalseQuarantine: v })} />
          <label className="flex items-center gap-2.5 cursor-pointer">
            <input type="checkbox" checked={w.denseFlagCredit}
              onChange={(e) => s.setWeights({ denseFlagCredit: e.target.checked })}
              className="accent-blue h-3.5 w-3.5" />
            <span className="text-xs text-ink-200">Dense flag credit</span>
            <span className="text-2xs text-ink-500">— pay in proportion to chain coverage</span>
          </label>

          <div className={`rounded-md border px-3 py-2.5 text-2xs leading-relaxed ${
            tie ? "border-green/30 bg-green-wash text-green" : "border-amber/40 bg-amber-wash text-amber"}`}>
            {tie ? (
              <>Degenerate-defender tie holds: at a 50/50 mix, quarantine-everything and
              allow-everything both score −{num(0.5 * w.wMiss, 2)}. Neither trivial strategy wins.</>
            ) : (
              <><TriangleAlert className="h-3 w-3 inline mr-1" />
              Tie broken. Quarantine-everything scores {num(0.5 * w.wDetect - 0.5 * w.wFalseQuarantine, 2)} vs
              allow-everything {num(-0.5 * w.wMiss, 2)} — one degenerate strategy is now optimal, and a
              defender can score well without learning to discriminate. Set w_false_quarantine = w_detect + w_miss to restore it.</>
            )}
          </div>
        </div>
      </Panel>

      <Panel title="Presentation" subtitle="Everything a presenter needs, in one place.">
        <div className="space-y-3">
          <div>
            <div className="label mb-1.5">Scenario</div>
            <select value={s.presetId} onChange={(e) => s.loadPreset(e.target.value)}
              className="w-full bg-ink-850 border border-ink-700/70 rounded-md text-xs px-2 h-9 focus-ring">
              {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
            <p className="text-2xs text-ink-500 mt-1.5">{PRESETS.find((p) => p.id === s.presetId)?.blurb}</p>
          </div>
          <div>
            <div className="label mb-1.5">Default defender</div>
            <select value={s.defenderId} onChange={(e) => s.setDefender(e.target.value)}
              className="w-full bg-ink-850 border border-ink-700/70 rounded-md text-xs px-2 h-9 focus-ring">
              {DEFENDERS.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
          <div className="flex items-center justify-between">
            <span className="label">Playback speed</span>
            <SegMap value={s.speed} onChange={s.setSpeed} options={[0.5, 1, 2, 4].map((v) => ({ value: v as any, label: `${v}×` }))} />
          </div>
          <div className="flex gap-2 pt-1">
            <Button variant="secondary" onClick={s.reset} className="flex-1"><RotateCcw className="h-3.5 w-3.5" />Reset demo</Button>
            <Button variant="danger" onClick={s.resetAll} className="flex-1">Factory reset</Button>
          </div>
        </div>
      </Panel>

      <Panel title="Connection" subtitle="Demo Mode is the default, by design.">
        <div className="space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-ink-300">Mode</span>
            <SegMap value={s.mode} onChange={s.setMode}
              options={[{ value: "demo", label: "DEMO" }, { value: "live", label: "LIVE" }]} />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-ink-300">Backend probe</span>
            <Badge tone={s.liveProbe === "ok" ? "green" : s.liveProbe === "unreachable" ? "red" : "neutral"}>
              {s.liveProbe}
            </Badge>
          </div>
          <p className="text-2xs text-ink-500 leading-relaxed pt-1 border-t hairline">
            The simulation engine — scenario generation, taint tracking, reward computation,
            every defender policy — runs entirely in this browser. No network, no API keys,
            no backend. Live Mode is an optional bridge to the Python implementation at
            <code className="font-mono text-ink-400"> 127.0.0.1:8000</code>; if it is
            unreachable the console falls back to Demo Mode and says so, rather than breaking.
          </p>
        </div>
      </Panel>
    </div>
  );
}

const num2 = (v: number) => v.toFixed(2);

function Slider({ label, value, min, max, step, onChange, fmt, hint }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; fmt: (v: number) => string; hint?: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-xs font-mono text-ink-200">{label}</span>
        <span className="text-xs tnum text-ink-100">{fmt(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-blue h-1 bg-ink-800 rounded-full appearance-none cursor-pointer" />
      {hint && <p className="text-2xs text-ink-500 mt-1 leading-snug">{hint}</p>}
    </div>
  );
}
