import { useState } from "react";
import { EyeOff, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { useStore } from "../../store/useStore";
import { Badge, Button, Dot, Panel, SegMap, Stat } from "../../components/ui";
import { EDGES, PIPELINE, PipelineNode, NodeStatus } from "../../lib/pipeline";
import { cx, num } from "../../lib/format";

const COLS = 7, ROWS = 3;
const NW = 138, NH = 62, GX = 176, GY = 92, PADX = 24, PADY = 20;
const W = PADX * 2 + (COLS - 1) * GX + NW;
const H = PADY * 2 + (ROWS - 1) * GY + NH;
const nx = (n: PipelineNode) => PADX + n.col * GX;
const ny = (n: PipelineNode) => PADY + n.row * GY;

const STATUS_COLOR: Record<NodeStatus, string> = {
  queued: "#3B4250", processing: "#3DD8E8", complete: "#3ED598",
  warning: "#F2B33D", error: "#F04A5E", skipped: "#2A2F3A",
};

export default function Simulation() {
  const s = useStore();
  const [sel, setSel] = useState<string | null>(null);
  const ep = s.episode;

  const node = sel ? PIPELINE.find((n) => n.id === sel)! : null;
  const done = PIPELINE.filter((n) => s.nodeStatus[n.id] === "complete").length;

  return (
    <div className="p-4 lg:p-5 space-y-4">
      <div className="panel px-4 py-3 flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5">
          {s.runState === "running"
            ? <Button size="sm" variant="secondary" onClick={s.pause}><Pause className="h-3 w-3" />Pause</Button>
            : <Button size="sm" variant="primary" onClick={s.run} disabled={ep?.ended}><Play className="h-3 w-3" />Run</Button>}
          <Button size="sm" variant="secondary" onClick={s.stepOnce} disabled={ep?.ended}><SkipForward className="h-3 w-3" />Step</Button>
          <Button size="sm" variant="ghost" onClick={() => s.loadPreset(s.presetId)}><RotateCcw className="h-3 w-3" />Restart</Button>
        </div>
        <span className="h-4 w-px bg-ink-700 mx-1" />
        <SegMap value={s.speed} onChange={s.setSpeed} options={[0.5, 1, 2, 4].map((v) => ({ value: v as any, label: `${v}×` }))} />
        <div className="ml-auto flex items-center gap-3 text-2xs text-ink-400">
          <span className="tnum">{done}/{PIPELINE.length} stages complete</span>
          <Badge tone={s.mode === "demo" ? "amber" : "green"}><Dot tone={s.mode === "demo" ? "amber" : "green"} />{s.mode.toUpperCase()}</Badge>
        </div>
      </div>

      <div className="grid xl:grid-cols-[1fr_320px] gap-4 items-start">
        <Panel title="Execution graph" subtitle="Stages map 1:1 onto the Python modules. Click any node to inspect it."
          bodyClass="p-3 overflow-x-auto">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[860px]" style={{ maxHeight: 340 }}>
            <defs>
              <marker id="ah" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill="#3B4250" />
              </marker>
            </defs>
            {EDGES.map(([a, b]) => {
              const A = PIPELINE.find((n) => n.id === a)!, B = PIPELINE.find((n) => n.id === b)!;
              const back = B.col < A.col;
              const x1 = nx(A) + (back ? NW / 2 : NW), y1 = ny(A) + NH / 2;
              const x2 = back ? nx(B) + NW / 2 : nx(B), y2 = ny(B) + NH / 2;
              const active = s.nodeStatus[a] === "complete" && s.nodeStatus[b] === "processing";
              const d = back
                ? `M${x1},${y1 - NH / 2} C${x1},${y1 - 60} ${x2},${y2 - 60} ${x2},${y2 - NH / 2}`
                : `M${x1},${y1} C${x1 + 40},${y1} ${x2 - 40},${y2} ${x2},${y2}`;
              return (
                <path key={`${a}-${b}`} d={d} fill="none" markerEnd="url(#ah)"
                  stroke={active ? "#3DD8E8" : "#2A2F3A"} strokeWidth={active ? 2 : 1.4}
                  strokeDasharray={active ? "6 6" : undefined}
                  className={active ? "animate-flow" : undefined} />
              );
            })}
            {PIPELINE.map((n) => {
              const st = s.nodeStatus[n.id];
              const c = STATUS_COLOR[st];
              const isActive = st === "processing";
              return (
                <g key={n.id} onClick={() => setSel(n.id)} className="cursor-pointer">
                  <rect x={nx(n)} y={ny(n)} width={NW} height={NH} rx="9"
                    fill={isActive ? "#0F2830" : "#111318"} stroke={c} strokeWidth={isActive ? 2 : 1.2}
                    className={cx(isActive && "animate-pulse-node", sel === n.id && "drop-shadow")} />
                  {n.internal && (
                    <rect x={nx(n)} y={ny(n)} width={NW} height={NH} rx="9" fill="none"
                      stroke="#9B8CFF" strokeWidth="1" strokeDasharray="3 3" opacity=".55" />
                  )}
                  <circle cx={nx(n) + 13} cy={ny(n) + 15} r="3.5" fill={c} />
                  <text x={nx(n) + 24} y={ny(n) + 19} className="fill-ink-100" style={{ fontSize: 11, fontWeight: 600 }}>
                    {n.label.length > 17 ? n.label.slice(0, 16) + "…" : n.label}
                  </text>
                  <text x={nx(n) + 13} y={ny(n) + 35} className="fill-ink-500" style={{ fontSize: 8.5, fontFamily: "ui-monospace, monospace" }}>
                    {n.module}
                  </text>
                  <text x={nx(n) + 13} y={ny(n) + 50} style={{ fontSize: 9, fill: c }}>{st}</text>
                </g>
              );
            })}
          </svg>
          <div className="flex flex-wrap items-center gap-3 mt-2 pt-2 border-t hairline text-2xs text-ink-500">
            {(["queued", "processing", "complete", "warning", "error"] as NodeStatus[]).map((k) => (
              <span key={k} className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: STATUS_COLOR[k] }} />{k}
              </span>
            ))}
            <span className="flex items-center gap-1.5 ml-auto">
              <EyeOff className="h-3 w-3 text-violet" /> dashed = ground truth, never observed by the defender
            </span>
          </div>
        </Panel>

        <div className="space-y-4">
          {node ? (
            <Panel title={node.label} subtitle={node.module} dense
              actions={<Button size="sm" variant="ghost" onClick={() => setSel(null)}>Clear</Button>}>
              <div className="space-y-3 text-xs">
                {node.internal && (
                  <div className="rounded-md border border-violet/30 bg-violet-wash px-2.5 py-2 text-2xs text-violet">
                    Env-internal. The defender never observes this stage — attack success is judged here, by data flow.
                  </div>
                )}
                <Field k="Purpose" v={node.purpose} />
                <Field k="Input" v={node.input} mono />
                <Field k="Processing" v={node.processing} />
                <Field k="Output" v={node.output} mono />
                <div className="flex items-center justify-between pt-2 border-t hairline">
                  <span className="label">Status</span>
                  <Badge tone={s.nodeStatus[node.id] === "complete" ? "green" : s.nodeStatus[node.id] === "processing" ? "blue" : "neutral"}>
                    {s.nodeStatus[node.id]}
                  </Badge>
                </div>
              </div>
            </Panel>
          ) : (
            <Panel title="Live metrics" dense subtitle="Driven by the same episode the graph is executing.">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Calls executed" value={ep?.history.length ?? 0} />
                <Stat label="Flow events" value={ep?.tracker.events.length ?? 0} tone="violet" />
                <Stat label="Live tags" value={ep?.tracker.artifacts.reduce((a, x) => a + x.tags.length, 0) ?? 0} tone="red" />
                <Stat label="Flags raised" value={ep?.flaggedSteps.length ?? 0} tone="amber" />
              </div>
              <div className="mt-3 space-y-2">
                <Stat label="R_blue (settled)" tone="blue"
                  value={ep?.breakdown ? num(ep.breakdown.rBlue, 2) : "—"}
                  sub={ep?.ended ? ep.endReason : "pending episode end"} />
              </div>
            </Panel>
          )}

          <Panel title="Stage queue" dense>
            <div className="space-y-1">
              {PIPELINE.map((n) => (
                <button key={n.id} onClick={() => setSel(n.id)}
                  className={cx("w-full flex items-center gap-2 rounded-md px-2 py-1.5 text-2xs transition-colors",
                    sel === n.id ? "bg-ink-800" : "hover:bg-ink-850")}>
                  <span className="h-1.5 w-1.5 rounded-full shrink-0" style={{ background: STATUS_COLOR[s.nodeStatus[n.id]] }} />
                  <span className="text-ink-200 truncate">{n.label}</span>
                  <span className="ml-auto text-ink-500">{s.nodeStatus[n.id]}</span>
                </button>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

const Field = ({ k, v, mono }: { k: string; v: string; mono?: boolean }) => (
  <div>
    <div className="label mb-1">{k}</div>
    <p className={cx("leading-relaxed text-ink-300", mono && "font-mono text-2xs")}>{v}</p>
  </div>
);
