import { useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Info } from "lucide-react";
import { useStore } from "../../store/useStore";
import { Badge, Panel, SegMap, Stat } from "../../components/ui";
import { BarsWithError, LineChart, RocCurve, rocPoints } from "../../components/viz";
import { DEFENDERS, defenderScore } from "../../lib/policies";
import { Episode } from "../../lib/engine";
import { sampleScenario } from "../../lib/scenarios";
import { makeRng } from "../../lib/rng";
import { makeScriptedAttacker, makeBenignRoller } from "../../lib/policies";
import { ALLOW } from "../../lib/domain";

/** Real 5-seed data from report/multiseed_baseline.json — the paired
 *  per-seed deltas that overturned the single-seed headline. */
const PAIRED = [0.090, -0.007, -0.203, 0.123, -0.357];
const CURVE = [0.720, 0.765, 0.740, 0.710];

/** Paired per-seed exploitability, lever − causal_monitor (docs/m11-gap.md). */
const LEVERS = [
  { name: "+ causal features", mean: -0.323, separated: true, seeds: [-0.553, -0.333, -0.053, -0.173, -0.503] },
  { name: "8 generations", mean: -0.116, separated: true, seeds: [-0.023, -0.170, -0.230, -0.087, -0.070] },
  { name: "dense flag credit", mean: 0.039, separated: false, seeds: [-0.063, -0.290, 0.277, 0.090, 0.183] },
  { name: "PFSP league", mean: 0.051, separated: false, seeds: [0.097, 0.020, -0.060, 0.080, 0.120] },
];

export default function Analysis() {
  const s = useStore();
  const [metric, setMetric] = useState<"expl" | "auroc" | "tpr">("expl");

  /** Score real decisions from freshly-rolled episodes so the ROC is computed,
   *  not drawn. Same rank-based AUC as arena/eval/metrics.py. */
  const roc = useMemo(() => {
    const rng = makeRng(99);
    const rows: { obs: any; label: number }[] = [];
    for (let i = 0; i < 40; i++) {
      const sc = sampleScenario(rng, s.gen, { adversarial: i % 2 === 0 });
      const ep = new Episode(sc, s.weights);
      const red = sc.adversarial ? makeScriptedAttacker(sc) : makeBenignRoller(sc, rng);
      while (!ep.ended) {
        if (ep.phase === "awaiting_red") ep.proposeTool(red());
        else { rows.push({ obs: ep.blueObservation(), label: 0 }); ep.adjudicate(ALLOW); }
      }
      const mal = new Set(ep.outcome!.maliciousSteps);
      rows.slice(-ep.history.length).forEach((r, j) => { r.label = mal.has(j) ? 1 : 0; });
    }
    return ["causal_monitor", "arena_blue", "single_shot", "static_allowlist"].map((id, i) => {
      const sc2 = rows.map((r) => defenderScore(id, r.obs));
      const { points, auc } = rocPoints(sc2, rows.map((r) => r.label));
      return { name: DEFENDERS.find((d) => d.id === id)!.name,
        color: ["#9B8CFF", "#3DD8E8", "#F2B33D", "#5A6273"][i], points, auc };
    });
  }, [s.gen, s.weights]);

  const key = metric === "expl" ? "expl" : metric === "auroc" ? "auroc" : "tpr";
  const sdKey = metric === "expl" ? "explSd" : metric === "auroc" ? "aurocSd" : "tprSd";

  return (
    <div className="p-4 lg:p-5 space-y-4">
      {/* The honest finding, first */}
      <div className="grid lg:grid-cols-2 gap-3">
        <div className="panel border-green/30 bg-green-wash/30 px-4 py-3 flex items-start gap-3">
          <CheckCircle2 className="h-4 w-4 text-green shrink-0 mt-0.5" />
          <div className="text-xs leading-relaxed">
            <span className="font-medium text-green">The gap closes — on every seed. </span>
            <span className="text-ink-300">
              Given the same causal features the CASPIAN-style baseline uses, the co-evolved
              defender reaches exploitability <span className="tnum text-ink-100">0.313</span> vs
              the baseline's <span className="tnum text-ink-100">0.636</span>, and it wins on
              all 5 of 5 seeds. The gap was representational, not a failure of co-evolution.
            </span>
          </div>
        </div>
        <div className="panel border-amber/30 bg-amber-wash/30 px-4 py-3 flex items-start gap-3">
          <AlertTriangle className="h-4 w-4 text-amber shrink-0 mt-0.5" />
          <div className="text-xs leading-relaxed">
            <span className="font-medium text-amber">Without those features, nothing is settled. </span>
            <span className="text-ink-300">
              Plain co-evolution and the causal monitor are statistically indistinguishable —
              the sign of the difference flips between seeds, so neither direction is claimed.
            </span>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-[1.3fr_1fr] gap-4 items-start">
        <Panel title="Leaderboard, 5 seeds" subtitle="Mean ± 1 SD. Brackets are the spread — read them before the bars."
          actions={<SegMap value={metric} onChange={setMetric}
            options={[{ value: "expl", label: "Exploitability" }, { value: "auroc", label: "AUROC" }, { value: "tpr", label: "TPR@5%" }]} />}>
          <BarsWithError max={1}
            rows={[...DEFENDERS]
              .sort((a, b) => (metric === "expl" ? a.stats.expl - b.stats.expl : b.stats[key as "auroc"] - a.stats[key as "auroc"]))
              .map((d) => ({
                label: d.name,
                value: d.stats[key as "expl"],
                sd: d.stats[sdKey as "explSd"],
                color: d.id === "arena_blue" ? "#3DD8E8" : d.id === "causal_monitor" ? "#9B8CFF" : "#5A6273",
              }))} />
          <p className="mt-4 text-2xs text-ink-500 flex items-start gap-1.5">
            <Info className="h-3 w-3 mt-px shrink-0" />
            {metric === "expl"
              ? "Lower is better. The co-evolved defender's ±0.274 spread is an order of magnitude wider than any static baseline's."
              : "Higher is better. Threshold-free, so these are not affected by the calibration step."}
          </p>
        </Panel>

        <Panel title="Paired per-seed difference" subtitle="arena_blue − causal_monitor on exploitability">
          <div className="space-y-1.5">
            {PAIRED.map((d, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-2xs font-mono text-ink-500 w-12">seed {i}</span>
                <div className="flex-1 relative h-5 bg-ink-850 rounded">
                  <div className="absolute inset-y-0 left-1/2 w-px bg-ink-600" />
                  <div className="absolute inset-y-1 rounded"
                    style={{
                      background: d < 0 ? "#3ED598" : "#F04A5E",
                      left: d < 0 ? `${50 + (d / 0.8) * 50}%` : "50%",
                      width: `${Math.abs(d / 0.8) * 50}%`,
                    }} />
                </div>
                <span className="text-2xs tnum w-14 text-right" style={{ color: d < 0 ? "#3ED598" : "#F04A5E" }}>
                  {d > 0 ? "+" : ""}{d.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t hairline space-y-1 text-2xs">
            <div className="flex justify-between"><span className="text-ink-500">Mean</span><span className="tnum text-ink-200">−0.071</span></div>
            <div className="flex justify-between"><span className="text-ink-500">Spread (SD)</span><span className="tnum text-ink-200">0.204</span></div>
            <div className="flex justify-between"><span className="text-ink-500">Verdict</span><Badge tone="amber">SIGN FLIPS</Badge></div>
          </div>
          <p className="mt-2.5 text-2xs text-ink-500 leading-relaxed">
            Three of five seeds favour co-evolution, two favour the causal monitor. A mean
            difference whose sign flips is not a result — so neither direction is claimed.
          </p>
        </Panel>
      </div>

      <div className="grid lg:grid-cols-2 gap-4 items-start">
        <Panel title="ROC — computed live" subtitle="40 freshly-rolled episodes, scored by each defender. Rank-based AUC.">
          <RocCurve curves={roc} />
        </Panel>

        <Panel title="Exploitability across self-play generations" subtitle="Does co-evolution actually drive the attack surface down?">
          <LineChart height={200} yMax={1}
            xTicks={["gen 0", "gen 1", "gen 2", "gen 3"]}
            series={[{ name: "exploitability", color: "#3DD8E8",
              points: CURVE.map((y, i) => ({ x: i, y })) }]} />
          <div className="mt-3 rounded-md border border-ink-700 bg-ink-850 px-3 py-2 text-2xs text-ink-400 leading-relaxed">
            Flat across four generations (0.720 → 0.710). Self-play is not yet driving
            exploitability down at this scale — stated rather than smoothed over.
          </div>
        </Panel>
      </div>

      <Panel title="Which levers actually worked" subtitle="Paired per-seed exploitability vs the causal monitor. Negative = co-evolved defender wins. A lever whose sign flips is not a result.">
        <div className="space-y-2.5">
          {LEVERS.map((l) => (
            <div key={l.name} className="flex items-center gap-3">
              <span className="text-xs text-ink-200 w-40 shrink-0">{l.name}</span>
              <div className="flex-1 flex items-center gap-1">
                {l.seeds.map((d, i) => (
                  <div key={i} className="flex-1 h-6 rounded relative bg-ink-850" title={`seed ${i}: ${d.toFixed(3)}`}>
                    <div className="absolute inset-y-1 rounded" style={{
                      background: d < 0 ? "#3ED598" : "#F04A5E",
                      left: d < 0 ? "0%" : "50%", right: d < 0 ? "50%" : "0%",
                      opacity: Math.min(1, 0.35 + Math.abs(d) * 1.2),
                    }} />
                  </div>
                ))}
              </div>
              <span className="text-2xs tnum w-14 text-right text-ink-300">{l.mean > 0 ? "+" : ""}{l.mean.toFixed(3)}</span>
              <Badge tone={l.separated ? "green" : "amber"}>{l.separated ? "5/5" : "flips"}</Badge>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat label="Seeds evaluated" value={5} sub="fresh self-play Blue trained per seed" />
        <Stat label="Widest spread" value="±0.274" tone="amber" sub="arena_blue exploitability" />
        <Stat label="Tightest spread" value="±0.024" tone="green" sub="static allow-list" />
        <Stat label="Stable result" value="5/5" tone="blue" sub="causal monitor beats allow-list" />
      </div>
    </div>
  );
}
