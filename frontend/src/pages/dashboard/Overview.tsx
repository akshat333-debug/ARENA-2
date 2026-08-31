import { Link } from "react-router-dom";
import { ArrowRight, Play, ShieldAlert, Swords, Timer } from "lucide-react";
import { useStore, PRESETS } from "../../store/useStore";
import { Badge, Button, Dot, Meter, Panel, Stat } from "../../components/ui";
import { Sparkline, StackedBar } from "../../components/viz";
import { DEFENDERS } from "../../lib/policies";
import { clock, num, pct } from "../../lib/format";

export default function Overview() {
  const s = useStore();
  const ep = s.episode;
  const h = s.history;
  const adv = h.filter((x) => x.adversarial);
  const landed = adv.filter((x) => x.completed).length;
  const stopped = adv.filter((x) => x.quarantined).length;
  const falsePos = h.filter((x) => !x.adversarial && x.quarantined).length;
  const def = DEFENDERS.find((d) => d.id === s.defenderId)!;

  return (
    <div className="p-4 lg:p-5 space-y-4">
      {/* Hero strip */}
      <div className="panel p-5 grid lg:grid-cols-[1fr_auto] gap-5 items-center grid-bg">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Badge tone="blue"><Dot tone="blue" pulse />Session active</Badge>
            <Badge tone={s.mode === "demo" ? "amber" : "green"}>{s.mode === "demo" ? "DEMO MODE — offline" : "LIVE"}</Badge>
          </div>
          <h1 className="text-xl font-semibold tracking-tight">
            {PRESETS.find((p) => p.id === s.presetId)?.name ?? "Custom scenario"}
          </h1>
          <p className="text-sm text-ink-400 mt-1.5 max-w-2xl leading-relaxed">
            {PRESETS.find((p) => p.id === s.presetId)?.blurb}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" onClick={() => { s.loadPreset(s.presetId); s.run(); }}>
            <Play className="h-3.5 w-3.5" />Run Demo
          </Button>
          <Link to="/app/arena"><Button variant="secondary"><Swords className="h-3.5 w-3.5" />Play the Arena</Button></Link>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat label="Episodes run" value={h.length} sub="this session" />
        <Stat label="Attacks landed" value={landed} tone="red"
          sub={adv.length ? `${pct(landed / adv.length, 0)} of ${adv.length} adversarial` : "no adversarial episodes yet"} />
        <Stat label="Stopped in flight" value={stopped} tone="green"
          sub={adv.length ? `${pct(stopped / adv.length, 0)} interception` : "—"} />
        <Stat label="False quarantines" value={falsePos} tone="amber"
          sub="benign sessions killed — the cost side of the ledger" />
      </div>

      <div className="grid lg:grid-cols-[1.4fr_1fr] gap-4 items-start">
        <Panel title="Defender reward over the session" subtitle="R_blue settles once per episode. Positive means it is discriminating, not blanket-blocking."
          actions={<Link to="/app/analysis"><Button size="sm" variant="ghost">Analysis <ArrowRight className="h-3 w-3" /></Button></Link>}>
          {h.length < 2
            ? <div className="py-10 text-center text-xs text-ink-500">Run at least two episodes to draw the curve.</div>
            : <>
              <Sparkline data={h.map((x) => x.rBlue)} tone="#3DD8E8" height={110} />
              <div className="grid grid-cols-3 gap-3 mt-4 pt-3 border-t hairline">
                <Stat label="Mean R_blue" value={num(h.reduce((a, x) => a + x.rBlue, 0) / h.length, 2)} tone="blue" />
                <Stat label="Mean R_red" value={num(h.reduce((a, x) => a + x.rRed, 0) / h.length, 2)} tone="red" />
                <Stat label="Latest" value={num(h[h.length - 1].rBlue, 2)} />
              </div>
            </>}
        </Panel>

        <div className="space-y-4">
          <Panel title="Active defender" dense subtitle={def.blurb}>
            <div className="space-y-2.5">
              <MetricLine label="AUROC" v={def.stats.auroc} sd={def.stats.aurocSd} tone="blue" />
              <MetricLine label="TPR @ 5% FPR" v={def.stats.tpr} sd={def.stats.tprSd} tone="green" />
              <MetricLine label="Exploitability" v={def.stats.expl} sd={def.stats.explSd} tone="red" invert />
            </div>
            <p className="mt-3 text-2xs text-ink-500 leading-relaxed">
              Means ± SD over 5 seeds. Exploitability is the primary metric — lower is better.
            </p>
          </Panel>

          <Panel title="Episode composition" dense>
            {h.length === 0
              ? <p className="text-2xs text-ink-500 py-2 text-center">No episodes yet.</p>
              : <>
                <StackedBar segments={[
                  { value: landed, color: "#F04A5E", label: "attack landed" },
                  { value: stopped, color: "#3ED598", label: "stopped" },
                  { value: adv.length - landed - stopped, color: "#F2B33D", label: "ran out of time" },
                  { value: h.length - adv.length, color: "#3B4250", label: "benign" },
                ]} />
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 mt-3 text-2xs">
                  <Legend c="#F04A5E" l="Attack landed" v={landed} />
                  <Legend c="#3ED598" l="Stopped in flight" v={stopped} />
                  <Legend c="#F2B33D" l="Timed out" v={adv.length - landed - stopped} />
                  <Legend c="#3B4250" l="Benign sessions" v={h.length - adv.length} />
                </div>
              </>}
          </Panel>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-4 items-start">
        <Panel title="Recent activity" subtitle="Tail of the execution log"
          actions={<Link to="/app/logs"><Button size="sm" variant="ghost">All logs <ArrowRight className="h-3 w-3" /></Button></Link>}
          bodyClass="p-0">
          <div className="max-h-[240px] overflow-y-auto divide-y divide-ink-800">
            {s.logs.slice(-9).reverse().map((l, i) => (
              <div key={i} className="flex items-start gap-2.5 px-4 py-2 text-2xs">
                <span className="font-mono text-ink-600 shrink-0">{clock(l.t).slice(0, 8)}</span>
                <Badge tone={l.level === "error" ? "red" : l.level === "warn" ? "amber" : l.level === "ok" ? "green" : "neutral"}>
                  {l.stage}
                </Badge>
                <span className="text-ink-300 leading-snug">{l.msg}</span>
              </div>
            ))}
            {!s.logs.length && <div className="px-4 py-8 text-center text-2xs text-ink-500">No activity yet.</div>}
          </div>
        </Panel>

        <Panel title="Current episode" subtitle={ep ? ep.scenario.task : "—"} dense>
          {!ep ? <p className="text-xs text-ink-500 py-4 text-center">No episode.</p> : (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-2xs">
                <span className="text-ink-500">Progress</span>
                <span className="tnum text-ink-300">{ep.stepIndex} / {ep.scenario.maxSteps} steps</span>
              </div>
              <Meter value={ep.stepIndex / ep.scenario.maxSteps} tone="blue" />
              <div className="grid grid-cols-2 gap-2 pt-1">
                <Stat label="Tools exposed" value={ep.scenario.registry.length} />
                <Stat label="Calls executed" value={ep.history.length} />
              </div>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {ep.scenario.adversarial
                  ? <Badge tone="red"><ShieldAlert className="h-3 w-3" />{ep.scenario.objective!.family.replace(/_/g, " ")}</Badge>
                  : <Badge tone="green">benign session</Badge>}
                <Badge tone="neutral"><Timer className="h-3 w-3" />{ep.endReason}</Badge>
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

function MetricLine({ label, v, sd, tone, invert }: {
  label: string; v: number; sd: number; tone: "blue" | "green" | "red"; invert?: boolean;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-xs text-ink-300">{label}</span>
        <span className="text-xs tnum text-ink-100">{num(v)} <span className="text-ink-500">±{num(sd)}</span></span>
      </div>
      <Meter value={invert ? 1 - v : v} tone={tone} />
    </div>
  );
}
const Legend = ({ c, l, v }: { c: string; l: string; v: number }) => (
  <span className="flex items-center gap-1.5 text-ink-400">
    <span className="h-2 w-2 rounded-sm shrink-0" style={{ background: c }} />{l}
    <span className="ml-auto tnum text-ink-200">{v}</span>
  </span>
);
