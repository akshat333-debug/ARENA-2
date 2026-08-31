/** Episode engine — mirrors arena/env.py.
 *
 *  One episode is one MCP session. Each step is two turns: Red proposes a tool
 *  call, Blue adjudicates it (allow / flag / quarantine). The episode ends on
 *  quarantine, on the objective completing, or at the step cap.
 */
import {
  ALLOW, FLAG, QUARANTINE, Scenario, ToolSpec, Verdict,
  isPrivileged, isSink, isSource, isUntrustedRead, propagates,
} from "./domain";
import { EpisodeOutcome, RewardBreakdown, RewardWeights, computeRewards, heuristicPlausibility } from "./rewards";
import { TaintTracker } from "./taint";

export interface ResolvedCall {
  stepIndex: number;
  tool: ToolSpec;
  verdict: Verdict;
  /** Ground truth, revealed only in the analyst views — never to a defender. */
  onAttackPath: boolean;
  tainted: boolean;
}

export type EpisodePhase = "idle" | "awaiting_red" | "awaiting_blue" | "ended";

export interface BlueObservation {
  domain: string;
  /** Observable metadata only — this is everything a defender may look at. */
  calls: {
    tool: ToolSpec; stepIndex: number; resolved: boolean; verdict: Verdict | null;
  }[];
  length: number;
}

export class Episode {
  readonly scenario: Scenario;
  readonly weights: RewardWeights;
  tracker = new TaintTracker();
  history: ResolvedCall[] = [];
  stepIndex = 0;
  pending: ToolSpec | null = null;
  flaggedSteps: number[] = [];
  quarantined = false;
  quarantineStep: number | null = null;
  caughtInFlight = false;
  completionStep: number | null = null;
  ended = false;
  outcome: EpisodeOutcome | null = null;
  breakdown: RewardBreakdown | null = null;

  constructor(scenario: Scenario, weights: RewardWeights) {
    this.scenario = scenario; this.weights = weights;
  }

  get phase(): EpisodePhase {
    if (this.ended) return "ended";
    return this.pending ? "awaiting_blue" : "awaiting_red";
  }

  /** What the defender is allowed to see. Deliberately excludes the task text,
   *  the objective and every byte of the taint graph. */
  blueObservation(): BlueObservation {
    const calls: BlueObservation["calls"] = this.history.map((r) => ({
      tool: r.tool, stepIndex: r.stepIndex, resolved: true, verdict: r.verdict as Verdict | null,
    }));
    if (this.pending) {
      calls.push({ tool: this.pending, stepIndex: this.stepIndex, resolved: false, verdict: null });
    }
    return { domain: this.scenario.domain, calls, length: calls.length };
  }

  proposeTool(tool: ToolSpec) {
    if (this.ended || this.pending) return;
    this.pending = tool;
  }

  adjudicate(verdict: Verdict) {
    if (this.ended || !this.pending) return;
    const tool = this.pending;
    const obj = this.scenario.objective;

    if (verdict === QUARANTINE) {
      this.quarantined = true;
      this.quarantineStep = this.stepIndex;
      this.caughtInFlight = this.tracker.wouldAdvanceAttack(obj, tool, this.stepIndex);
      this.pending = null;
      this.end();
      return;
    }

    const before = this.tracker.artifacts.length;
    this.tracker.step(tool, this.stepIndex);
    const tainted = this.tracker.artifacts.length > before &&
      (this.tracker.artifacts.at(-1)?.tags.length ?? 0) > 0;

    this.history.push({
      stepIndex: this.stepIndex, tool, verdict,
      onAttackPath: false, tainted,
    });
    if (verdict === FLAG) this.flaggedSteps.push(this.stepIndex);
    if (obj && this.completionStep === null && this.tracker.objectiveCompleted(obj)) {
      this.completionStep = this.stepIndex;
    }
    this.stepIndex += 1;
    this.pending = null;
    if (this.completionStep !== null || this.stepIndex >= this.scenario.maxSteps) this.end();
  }

  private end() {
    if (this.ended) return;
    this.ended = true;
    const obj = this.scenario.objective;
    // On a completed attack the satisfying event's path is ground truth. When
    // the defender stopped it there is no such event, so fall back to the chain
    // that was in flight.
    const malicious = obj
      ? (this.tracker.attackPath(obj).length ? this.tracker.attackPath(obj) : this.tracker.liveChainSteps(obj))
      : [];
    const mset = new Set(malicious);
    this.history = this.history.map((h) => ({ ...h, onAttackPath: mset.has(h.stepIndex) }));

    this.outcome = {
      adversarial: this.scenario.adversarial,
      objectiveCompleted: this.completionStep !== null,
      nSteps: this.history.length,
      plausibility: heuristicPlausibility(this.history.map((h) => h.tool)),
      quarantined: this.quarantined,
      flaggedSteps: [...this.flaggedSteps],
      maliciousSteps: malicious,
      caughtInFlight: this.caughtInFlight,
    };
    this.breakdown = computeRewards(this.outcome, this.weights);
  }

  /** Human-readable reason the episode ended. */
  get endReason(): string {
    if (!this.ended) return "running";
    if (this.quarantined) return this.caughtInFlight ? "quarantined in flight" : "quarantined (no evidence)";
    if (this.completionStep !== null) return "objective completed";
    return "step cap reached";
  }
}

/** Observable per-call risk signals — the same view every defender gets. */
export function callSignals(tool: ToolSpec) {
  return {
    source: isSource(tool), sink: isSink(tool), propagates: propagates(tool),
    untrusted: isUntrustedRead(tool), privileged: isPrivileged(tool),
  };
}

export { ALLOW, FLAG, QUARANTINE };
