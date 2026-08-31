import { CheckCircle2, Download, ShieldAlert, ShieldCheck, TriangleAlert } from "lucide-react";
import { useStore } from "../../store/useStore";
import { Badge, Button, EmptyState, Panel, Stat } from "../../components/ui";
import { Gauge } from "../../components/viz";
import { cx, num, pct } from "../../lib/format";

export default function Results() {
  const s = useStore();
  const ep = s.episode;

  if (!ep || !ep.ended || !ep.outcome) {
    return (
      <div className="p-6">
        <EmptyState
          title={ep ? "Episode still running" : "No episode loaded"}
          body={ep ? "Results are produced when the session terminates — on quarantine, objective completion, or the step cap."
                   : "Load a scenario from the sidebar to begin."}
          action={<Button variant="primary" onClick={() => { s.loadPreset(s.presetId); s.run(); }}>Run demo</Button>} />
      </div>
    );
  }

  const o = ep.outcome, b = ep.breakdown!;
  const defended = !o.objectiveCompleted && o.adversarial;
  const falsePos = o.quarantined && !o.adversarial;
  const verdict = falsePos ? "false-positive" : defended ? "defended" : o.adversarial ? "breached" : "clean";
  const V = {
    defended: { tone: "green", icon: ShieldCheck, head: "Attack stopped in flight",
      body: "The chain was interrupted before sensitive data crossed the session boundary." },
    breached: { tone: "red", icon: ShieldAlert, head: "Attack completed",
      body: "Sensitive data reached an external sink. The defender did not intervene in time." },
    "false-positive": { tone: "amber", icon: TriangleAlert, head: "Benign session quarantined",
      body: "No attack was present. This is the cost side of the ledger — the reason blanket paranoia is not a strategy." },
    clean: { tone: "green", icon: CheckCircle2, head: "Legitimate session completed",
      body: "Benign work ran to completion without interference." },
  }[verdict as "defended"];

  const exportReport = () => {
    const rep = {
      scenario: ep.scenario.id, domain: ep.scenario.domain,
      adversarial: o.adversarial, family: ep.scenario.objective?.family ?? null,
      verdict, endReason: ep.endReason,
      objectiveCompleted: o.objectiveCompleted, caughtInFlight: ep.caughtInFlight,
      steps: o.nSteps, maliciousSteps: o.maliciousSteps, flaggedSteps: o.flaggedSteps,
      rewards: { red: b.rRed, blue: b.rBlue },
      calls: ep.history.map((h) => ({ step: h.stepIndex, tool: h.tool.name, verdict: h.verdict, onAttackPath: h.onAttackPath })),
    };
    const blob = new Blob([JSON.stringify(rep, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${ep.scenario.id}-report.json`; a.click(); URL.revokeObjectURL(a.href);
    s.log("ok", "export", `Exported case report for ${ep.scenario.id}.`);
  };

  const precision = o.flaggedSteps.length
    ? o.flaggedSteps.filter((f) => o.maliciousSteps.includes(f)).length / o.flaggedSteps.length : null;
  const recall = o.maliciousSteps.length
    ? o.flaggedSteps.filter((f) => o.maliciousSteps.includes(f)).length / o.maliciousSteps.length : null;

  return (
    <div className="p-4 lg:p-5 space-y-4">
      <div className={cx("panel p-5 flex flex-wrap items-start gap-5 border-l-2",
        V.tone === "green" ? "border-l-green" : V.tone === "red" ? "border-l-red" : "border-l-amber")}>
        <V.icon className={cx("h-8 w-8 shrink-0",
          V.tone === "green" ? "text-green" : V.tone === "red" ? "text-red" : "text-amber")} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <Badge tone={V.tone as any}>{verdict}</Badge>
            <span className="text-2xs font-mono text-ink-500">{ep.scenario.id}</span>
          </div>
          <h1 className="text-xl font-semibold tracking-tight">{V.head}</h1>
          <p className="text-sm text-ink-400 mt-1.5 max-w-2xl leading-relaxed">{V.body}</p>
        </div>
        <Button variant="secondary" onClick={exportReport}><Download className="h-3.5 w-3.5" />Export report</Button>
      </div>

      <div className="grid lg:grid-cols-[1fr_320px] gap-4 items-start">
        <div className="space-y-4">
          <Panel title="What the system found" subtitle="Reconstructed from the taint graph — the ground truth the defender never saw.">
            <div className="space-y-2.5">
              {ep.scenario.adversarial ? (
                <>
                  <Finding k="Attack family" v={ep.scenario.objective!.family.replace(/_/g, " ")} />
                  <Finding k="Objective" v={ep.scenario.objective!.kind.replace(/_/g, " ")} />
                  <Finding k="Required chain length" v={`${ep.scenario.objective!.minChainLen} calls`} />
                  <Finding k="Designated source" v={ep.scenario.objective!.sourceTools.join(", ")} mono />
                  <Finding k="Malicious steps" v={o.maliciousSteps.length ? o.maliciousSteps.join(", ") : "none reached"} mono />
                  <Finding k="Boundary crossings" v={`${ep.tracker.events.filter((e) => e.kind !== "benign_sink").length} flow event(s)`} />
                </>
              ) : (
                <Finding k="Session type" v="benign — no objective was ever present" />
              )}
              <Finding k="Termination" v={ep.endReason} />
            </div>
          </Panel>

          <Panel title="Call-by-call reconstruction" bodyClass="p-0">
            <div className="divide-y divide-ink-800">
              {ep.history.map((h) => (
                <div key={h.stepIndex} className={cx("flex items-center gap-3 px-4 py-2 text-xs",
                  h.onAttackPath && "bg-red-wash/25")}>
                  <span className="font-mono text-2xs text-ink-600 w-5 tnum">{h.stepIndex}</span>
                  <span className="font-mono text-ink-100 flex-1 truncate">{h.tool.name}</span>
                  <span className="text-2xs text-ink-500">{h.tool.sideEffect}</span>
                  {h.onAttackPath && <Badge tone="red">on path</Badge>}
                  <Badge tone={h.verdict === 1 ? "amber" : "neutral"}>{h.verdict === 1 ? "flag" : "allow"}</Badge>
                </div>
              ))}
              {!ep.history.length && <div className="px-4 py-8 text-center text-2xs text-ink-500">No calls executed — quarantined at step 0.</div>}
            </div>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title="Confidence" dense>
            <div className="flex items-center justify-around py-2">
              <Gauge value={o.objectiveCompleted ? 0.97 : ep.caughtInFlight ? 0.91 : 0.55}
                label="verdict" tone={V.tone === "green" ? "#3ED598" : V.tone === "red" ? "#F04A5E" : "#F2B33D"} />
              <div className="space-y-2 text-2xs">
                <div><div className="label">Evidence basis</div>
                  <div className="text-ink-200 mt-0.5">{ep.tracker.events.length} flow event(s)</div></div>
                <div><div className="label">Chain observed</div>
                  <div className="text-ink-200 mt-0.5">{o.maliciousSteps.length} step(s)</div></div>
              </div>
            </div>
          </Panel>

          <Panel title="Defender performance" dense subtitle="This episode only.">
            <div className="grid grid-cols-2 gap-2">
              <Stat label="R_blue" value={num(b.rBlue, 2)} tone={b.rBlue >= 0 ? "blue" : "red"} />
              <Stat label="R_red" value={num(b.rRed, 2)} tone="red" />
            </div>
            <div className="mt-3 space-y-2 text-2xs">
              <Line k="Flag precision" v={precision === null ? "—" : pct(precision, 0)} />
              <Line k="Chain coverage" v={recall === null ? "—" : pct(recall, 0)} />
              <Line k="Caught in flight" v={ep.caughtInFlight ? "yes" : "no"} />
              <Line k="Steps used" v={`${o.nSteps} / ${ep.scenario.maxSteps}`} />
            </div>
          </Panel>

          <Panel title="Recommended action" dense>
            <p className="text-xs leading-relaxed text-ink-300">
              {verdict === "breached"
                ? "Revoke the credentials read at the source step and audit the sink endpoint. Then re-run this scenario against a trajectory-aware defender — a per-call rule table cannot catch this shape."
                : verdict === "defended"
                ? "No action required. Preserve this episode as a regression case: it is a chain a per-call defender would have allowed."
                : verdict === "false-positive"
                ? "Review the quarantine threshold. This session did legitimate work and was killed; the false-positive weight is deliberately set so this costs as much as a miss."
                : "No action required."}
            </p>
          </Panel>
        </div>
      </div>
    </div>
  );
}

const Finding = ({ k, v, mono }: { k: string; v: string; mono?: boolean }) => (
  <div className="flex items-start justify-between gap-4 text-xs">
    <span className="text-ink-500 shrink-0">{k}</span>
    <span className={cx("text-ink-200 text-right", mono && "font-mono text-2xs")}>{v}</span>
  </div>
);
const Line = ({ k, v }: { k: string; v: string }) => (
  <div className="flex justify-between"><span className="text-ink-500">{k}</span><span className="tnum text-ink-200">{v}</span></div>
);
