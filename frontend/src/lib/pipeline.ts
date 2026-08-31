/** The ARENA execution pipeline — the real architecture, as a node graph.
 *  Stages map 1:1 onto the Python modules so the Simulation view is a picture
 *  of the actual system, not an illustration of a generic one.
 */
export type NodeStatus = "queued" | "processing" | "complete" | "warning" | "error" | "skipped";

export interface PipelineNode {
  id: string;
  label: string;
  module: string;
  purpose: string;
  input: string;
  processing: string;
  output: string;
  /** Ground-truth stages the defender must never observe. */
  internal?: boolean;
  col: number;
  row: number;
}

export const PIPELINE: PipelineNode[] = [
  { id: "scenario", label: "Scenario Generator", module: "arena/scenarios.py", col: 0, row: 1,
    purpose: "Draw an episode: domain, cover task, attack family, and whether this session is adversarial at all.",
    input: "Seed + ScenarioConfig (families, adversarial ratio, tool budget)",
    processing: "Family is chosen before the registry is sampled, so a winning chain can be guaranteed to exist.",
    output: "Scenario { domain, task, adversarial, objective }" },
  { id: "registry", label: "Registry Sampler", module: "arena/tools.py", col: 1, row: 1,
    purpose: "Expose a tool surface for this session.",
    input: "Domain + tool count",
    processing: "Guarantees a source → propagator → sink chain is present. No episode is unwinnable by construction.",
    output: "ToolRegistry (6-10 tools with declared side effects)" },
  { id: "red", label: "Red Policy", module: "arena/policies.py", col: 2, row: 0,
    purpose: "Propose the next tool call.",
    input: "Task, objective, registry, and the defender's past verdicts",
    processing: "Scores tools by embedding against a context vector — so one attacker generalises across registries and adapts to the defender it faces.",
    output: "Tool index" },
  { id: "blue", label: "Blue Policy", module: "arena/policies.py", col: 3, row: 1,
    purpose: "Adjudicate the proposed call: allow, flag, or quarantine.",
    input: "Observable call metadata only — side effect, category, sensitivity, position, past verdicts",
    processing: "GRU over the call sequence: trajectory-level by construction. Never sees the task, the objective, or the taint graph.",
    output: "Verdict ∈ {allow, flag, quarantine}" },
  { id: "taint", label: "Taint Tracker", module: "arena/taint.py", col: 4, row: 2, internal: true,
    purpose: "Ground truth. Decide whether the attack actually completed.",
    input: "The executed call sequence",
    processing: "Sources mint SENSITIVE/UNTRUSTED tags; transform and exec carry them; a sink crossing the boundary records a FlowEvent. Chain length gates completion.",
    output: "FlowEvents + attack path" },
  { id: "reward", label: "Reward Engine", module: "arena/rewards.py", col: 5, row: 1, internal: true,
    purpose: "Price the episode for both sides.",
    input: "EpisodeOutcome + RewardConfig weights",
    processing: "Asymmetric. Detection credit is gated on evidence; the false-quarantine weight makes blanket paranoia score exactly as badly as blanket permissiveness.",
    output: "R_red, R_blue" },
  { id: "outcome", label: "Outcome / Metrics", module: "arena/eval/", col: 6, row: 1,
    purpose: "Aggregate into the metrics that are actually reported.",
    input: "Episode outcomes across the run",
    processing: "Exploitability (fresh best-response attacker vs the frozen defender), AUROC, TPR@5%FPR — all rank-based and checked against sklearn.",
    output: "Leaderboard row" },
];

export const EDGES: [string, string][] = [
  ["scenario", "registry"], ["registry", "red"], ["red", "blue"],
  ["blue", "taint"], ["taint", "reward"], ["reward", "outcome"],
  ["blue", "red"],
];

export const nodeById = (id: string) => PIPELINE.find((n) => n.id === id)!;
