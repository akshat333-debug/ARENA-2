import {
  Bot, Crown, Eye, EyeOff, Hand, RefreshCw, ShieldCheck, ShieldQuestion, ShieldX, Swords, Zap,
} from "lucide-react";
import { useStore, ALLOW, FLAG, QUARANTINE } from "../../store/useStore";
import { Badge, Button, Dot, EmptyState, Meter, Panel, SegMap, Stat } from "../../components/ui";
import { Sparkline } from "../../components/viz";
import { BLUE_POLICIES, DEFENDERS, defenderScore, sequenceFeatures } from "../../lib/policies";
import { ToolSpec, isPrivileged, isSink, isSource, isUntrustedRead, propagates } from "../../lib/domain";
import { cx, num } from "../../lib/format";

export default function Arena() {
  const s = useStore();
  const ep = s.episode;

  // Autoplay is driven globally by <RunDriver/> in the Shell.

  if (!ep) return <div className="p-6"><EmptyState title="No episode loaded" action={<Button onClick={() => s.loadPreset("std")}>Load scenario</Button>} /></div>;

  const obs = ep.blueObservation();
  const risk = defenderScore(s.defenderId, obs);
  const feats = sequenceFeatures(obs);
  const recommended = (BLUE_POLICIES[s.defenderId] ?? BLUE_POLICIES.causal_monitor)(obs);
  const blueTurn = ep.phase === "awaiting_blue";
  const redTurn = ep.phase === "awaiting_red";

  return (
    <div className="p-4 lg:p-5 space-y-4">
      {/* Control strip */}
      <div className="panel px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-2">
          <span className="label">Red</span>
          <SegMap value={s.redControl} onChange={s.setRedControl}
            options={[
              { value: "human", label: <><Hand className="h-3 w-3 inline mr-1" />You</> },
              { value: "scripted", label: "Scripted" },
              { value: "adaptive", label: "Adaptive" },
            ]} />
        </div>
        <div className="flex items-center gap-2">
          <span className="label">Blue</span>
          <SegMap value={s.blueControl} onChange={s.setBlueControl}
            options={[
              { value: "human", label: <><Hand className="h-3 w-3 inline mr-1" />You</> },
              { value: "policy", label: <><Bot className="h-3 w-3 inline mr-1" />Policy</> },
            ]} />
        </div>
        {s.blueControl === "policy" && (
          <select value={s.defenderId} onChange={(e) => s.setDefender(e.target.value)}
            className="bg-ink-850 border border-ink-700/70 rounded-md text-xs px-2 h-7 focus-ring">
            {DEFENDERS.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
        )}
        <div className="ml-auto flex items-center gap-2">
          <SegMap value={s.speed} onChange={s.setSpeed}
            options={[0.5, 1, 2, 4].map((v) => ({ value: v as any, label: `${v}×` }))} />
          <Button size="sm" variant="secondary" onClick={s.stepOnce} disabled={ep.ended}>Step</Button>
          {s.runState === "running"
            ? <Button size="sm" variant="secondary" onClick={s.pause}>Pause</Button>
            : <Button size="sm" variant="primary" onClick={s.run} disabled={ep.ended}><Zap className="h-3 w-3" />Auto</Button>}
          <Button size="sm" variant="ghost" onClick={() => s.newEpisode()}><RefreshCw className="h-3 w-3" />New episode</Button>
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_360px] gap-4 items-start">
        {/* Board */}
        <div className="space-y-4">
          <Panel
            title={<span className="flex items-center gap-2"><Swords className="h-4 w-4 text-ink-400" />Session ledger</span>}
            subtitle={ep.scenario.task}
            actions={<Badge tone={ep.ended ? "green" : redTurn ? "red" : "blue"}>
              <Dot tone={ep.ended ? "green" : redTurn ? "red" : "blue"} pulse={!ep.ended} />
              {ep.ended ? ep.endReason : redTurn ? "Red to move" : "Blue to adjudicate"}
            </Badge>}>
            <div className="space-y-1.5">
              {ep.history.length === 0 && !ep.pending && (
                <p className="text-xs text-ink-500 py-6 text-center">No calls yet — Red moves first.</p>
              )}
              {ep.history.map((h) => <CallRow key={h.stepIndex} step={h.stepIndex} tool={h.tool}
                verdict={h.verdict} onPath={h.onAttackPath} revealed={ep.ended} />)}
              {ep.pending && <CallRow step={ep.stepIndex} tool={ep.pending} verdict={null} pending />}
            </div>
          </Panel>

          {/* Red's move */}
          {s.redControl === "human" && !ep.ended && (
            <Panel title="Red — choose the next tool call" dense
              subtitle={ep.scenario.adversarial
                ? `Objective: ${ep.scenario.objective!.kind.replace(/_/g, " ")} · chain ≥ ${ep.scenario.objective!.minChainLen} calls`
                : "Benign session — no objective. Try to look like normal work."}>
              <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-1.5">
                {ep.scenario.registry.map((t) => (
                  <ToolButton key={t.name} tool={t} disabled={!redTurn}
                    objective={ep.scenario.objective}
                    onClick={() => s.propose(t)} />
                ))}
              </div>
            </Panel>
          )}

          {/* Blue's move */}
          {s.blueControl === "human" && blueTurn && (
            <Panel title="Blue — adjudicate the pending call" dense
              subtitle="You see observable metadata only. No task text, no objective, no taint graph.">
              <div className="grid sm:grid-cols-3 gap-2">
                <VerdictButton icon={ShieldCheck} tone="green" label="Allow" hint="Executes. Costs nothing if benign."
                  rec={recommended === ALLOW} onClick={() => s.adjudicate(ALLOW)} />
                <VerdictButton icon={ShieldQuestion} tone="amber" label="Flag" hint="Executes but is recorded. Credit if on-chain, cost if not."
                  rec={recommended === FLAG} onClick={() => s.adjudicate(FLAG)} />
                <VerdictButton icon={ShieldX} tone="red" label="Quarantine" hint="Ends the session. Heavy penalty on a benign one."
                  rec={recommended === QUARANTINE} onClick={() => s.adjudicate(QUARANTINE)} />
              </div>
              <div className="mt-3 pt-3 border-t hairline grid sm:grid-cols-2 gap-3">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="label">Observable risk</span>
                    <span className="text-xs tnum text-ink-200">{num(risk, 2)}</span>
                  </div>
                  <Meter value={risk} tone={risk > 0.6 ? "red" : risk > 0.35 ? "amber" : "green"} />
                </div>
                <div className="space-y-1">
                  <Signal on={feats.hasSensitiveRead} label="sensitive read seen" />
                  <Signal on={feats.sensitiveReadBeforeSink} label="sensitive read BEFORE a sink" strong />
                  <Signal on={feats.untrustedReadBeforePrivileged} label="untrusted read before privileged call" strong />
                  <Signal on={feats.launderDepth > 0} label={`${feats.launderDepth} laundering step(s)`} />
                </div>
              </div>
            </Panel>
          )}
        </div>

        {/* Scoreboard */}
        <div className="space-y-4">
          <Panel title="Scoreboard" dense subtitle="Reward is settled when the episode ends.">
            <div className="grid grid-cols-2 gap-2">
              <Stat label="R_red" tone="red" value={ep.breakdown ? num(ep.breakdown.rRed, 2) : "—"} />
              <Stat label="R_blue" tone="blue" value={ep.breakdown ? num(ep.breakdown.rBlue, 2) : "—"} />
            </div>
            {ep.breakdown && (
              <div className="mt-3 space-y-1 text-2xs">
                {([["evade", ep.breakdown.redEvade], ["plausibility", ep.breakdown.redPlausibility],
                  ["step cost", -ep.breakdown.redStepCost], ["detect", ep.breakdown.blueDetect],
                  ["on-chain flag", ep.breakdown.blueFlag], ["miss", -ep.breakdown.blueMiss],
                  ["false quarantine", -ep.breakdown.blueFalseQuarantine],
                  ["off-chain flags", -ep.breakdown.blueFalseFlag]] as [string, number][])
                  .filter(([, v]) => v !== 0)
                  .map(([k, v]) => (
                    <div key={k as string} className="flex justify-between">
                      <span className="text-ink-500">{k as string}</span>
                      <span className={cx("tnum", (v as number) > 0 ? "text-green" : "text-red-soft")}>
                        {(v as number) > 0 ? "+" : ""}{num(v as number, 2)}
                      </span>
                    </div>
                  ))}
              </div>
            )}
          </Panel>

          <Panel title="Match history" dense subtitle={`${s.history.length} episode(s) this session`}>
            {s.history.length === 0
              ? <p className="text-2xs text-ink-500 py-3 text-center">Finish an episode to build the curve.</p>
              : (<>
                <Sparkline data={s.history.map((h) => h.rBlue)} tone="#3DD8E8" height={40} />
                <div className="grid grid-cols-3 gap-2 mt-3 text-center">
                  <MiniStat label="Attacks landed" value={s.history.filter((h) => h.completed).length} tone="text-red-soft" />
                  <MiniStat label="Stopped" value={s.history.filter((h) => h.quarantined && h.adversarial).length} tone="text-green" />
                  <MiniStat label="False quarantines" value={s.history.filter((h) => h.quarantined && !h.adversarial).length} tone="text-amber" />
                </div>
              </>)}
          </Panel>

          <Panel title="Ground truth" dense
            subtitle="Env-internal. Revealed here for the analyst; never shown to the defender."
            actions={<Badge tone={ep.ended ? "violet" : "neutral"}>{ep.ended ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}</Badge>}>
            {!ep.ended
              ? <p className="text-2xs text-ink-500 py-2">Sealed until the episode ends — so you cannot play Blue with the answers.</p>
              : (
                <div className="space-y-2 text-2xs">
                  <Row k="Objective" v={ep.scenario.objective ? ep.scenario.objective.kind.replace(/_/g, " ") : "none (benign)"} />
                  <Row k="Completed" v={ep.outcome!.objectiveCompleted ? "YES — data crossed the boundary" : "no"} />
                  <Row k="Malicious steps" v={ep.outcome!.maliciousSteps.length ? ep.outcome!.maliciousSteps.join(", ") : "—"} />
                  <Row k="Caught in flight" v={ep.caughtInFlight ? "yes (detection credit earned)" : "no"} />
                  <Row k="Flow events" v={String(ep.tracker.events.length)} />
                </div>
              )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function CallRow({ step, tool, verdict, onPath, pending, revealed }: {
  step: number; tool: ToolSpec; verdict: number | null;
  onPath?: boolean; pending?: boolean; revealed?: boolean;
}) {
  const v = verdict === ALLOW ? ["allow", "neutral"] : verdict === FLAG ? ["flag", "amber"]
    : verdict === QUARANTINE ? ["quarantine", "red"] : ["pending", "blue"];
  return (
    <div className={cx("relative flex items-center gap-3 rounded-lg border px-3 py-2 transition-colors",
      pending ? "border-blue/50 bg-blue-wash animate-fade-up" : "border-ink-700/70 bg-ink-850/60")}>
      {revealed && onPath && <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-red" title="on the attack path (ground truth)" />}
      <span className="font-mono text-2xs text-ink-500 w-5 tnum">{step}</span>
      <div className="min-w-0 flex-1">
        <div className="font-mono text-xs text-ink-100 truncate">{tool.name}</div>
        <div className="text-2xs text-ink-500">{tool.sideEffect} · sens {tool.sensitivity}</div>
      </div>
      <div className="flex items-center gap-1">
        {isSource(tool) && <Badge tone="red">source</Badge>}
        {propagates(tool) && <Badge tone="violet">carries</Badge>}
        {isSink(tool) && <Badge tone="amber">sink</Badge>}
        {isUntrustedRead(tool) && <Badge tone="violet">untrusted</Badge>}
      </div>
      <Badge tone={v[1] as any} mono>{v[0]}</Badge>
    </div>
  );
}

function ToolButton({ tool, onClick, disabled, objective }: {
  tool: ToolSpec; onClick: () => void; disabled?: boolean; objective: any;
}) {
  const isObjSource = objective?.sourceTools?.includes(tool.name);
  const isObjSink = objective?.sinkTools?.includes(tool.name);
  return (
    <button onClick={onClick} disabled={disabled} title={tool.description}
      className={cx("text-left rounded-lg border px-2.5 py-2 transition-colors focus-ring disabled:opacity-40",
        "border-ink-700/70 bg-ink-850 hover:bg-ink-800 hover:border-ink-600")}>
      <div className="font-mono text-2xs text-ink-100 truncate">{tool.name}</div>
      <div className="flex items-center gap-1 mt-1 flex-wrap">
        <span className="text-2xs text-ink-500">{tool.sideEffect}</span>
        {isObjSource && <Badge tone="red">obj source</Badge>}
        {isObjSink && <Badge tone="amber">obj sink</Badge>}
        {isPrivileged(tool) && !isObjSource && !isObjSink && <Badge tone="neutral">priv</Badge>}
      </div>
    </button>
  );
}

function VerdictButton({ icon: Icon, tone, label, hint, rec, onClick }: {
  icon: any; tone: "green" | "amber" | "red"; label: string; hint: string; rec?: boolean; onClick: () => void;
}) {
  const c = { green: "hover:border-green/50 hover:bg-green-wash", amber: "hover:border-amber/50 hover:bg-amber-wash", red: "hover:border-red/50 hover:bg-red-wash" }[tone];
  const ic = { green: "text-green", amber: "text-amber", red: "text-red" }[tone];
  return (
    <button onClick={onClick}
      className={cx("relative text-left rounded-lg border border-ink-700/70 bg-ink-850 px-3 py-2.5 transition-colors focus-ring", c)}>
      {rec && <span className="absolute top-2 right-2"><Badge tone="blue"><Crown className="h-2.5 w-2.5" />policy</Badge></span>}
      <Icon className={cx("h-4 w-4 mb-1.5", ic)} />
      <div className="text-sm font-medium">{label}</div>
      <div className="text-2xs text-ink-500 mt-0.5 leading-snug">{hint}</div>
    </button>
  );
}

const Signal = ({ on, label, strong }: { on: boolean; label: string; strong?: boolean }) => (
  <div className="flex items-center gap-1.5 text-2xs">
    <Dot tone={on ? (strong ? "red" : "amber") : "neutral"} />
    <span className={on ? (strong ? "text-red-soft" : "text-ink-200") : "text-ink-500"}>{label}</span>
  </div>
);
const MiniStat = ({ label, value, tone }: { label: string; value: number; tone: string }) => (
  <div><div className={cx("text-lg font-semibold tnum", tone)}>{value}</div><div className="text-2xs text-ink-500">{label}</div></div>
);
const Row = ({ k, v }: { k: string; v: string }) => (
  <div className="flex justify-between gap-3"><span className="text-ink-500">{k}</span><span className="text-ink-200 text-right">{v}</span></div>
);
