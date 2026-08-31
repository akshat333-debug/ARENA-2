import { useState } from "react";
import { ArrowRight, EyeOff, Minus, Plus } from "lucide-react";
import { Badge, Button, Panel } from "../../components/ui";
import { PIPELINE, EDGES } from "../../lib/pipeline";
import { FAMILIES } from "../../lib/domain";
import { cx } from "../../lib/format";

export default function Workflow() {
  const [sel, setSel] = useState(PIPELINE[0].id);
  const [zoom, setZoom] = useState(1);
  const node = PIPELINE.find((n) => n.id === sel)!;
  const upstream = EDGES.filter(([, b]) => b === sel).map(([a]) => a);
  const downstream = EDGES.filter(([a]) => a === sel).map(([, b]) => b);

  return (
    <div className="p-4 lg:p-5 space-y-4">
      <Panel title="System architecture" subtitle="Seven stages. Click any to inspect its contract."
        actions={<div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" onClick={() => setZoom((z) => Math.max(0.7, z - 0.15))}><Minus className="h-3 w-3" /></Button>
          <span className="text-2xs tnum text-ink-400 w-10 text-center">{Math.round(zoom * 100)}%</span>
          <Button size="sm" variant="ghost" onClick={() => setZoom((z) => Math.min(1.6, z + 0.15))}><Plus className="h-3 w-3" /></Button>
        </div>}
        bodyClass="p-4 overflow-auto">
        <div className="flex flex-wrap gap-2 origin-top-left transition-transform" style={{ transform: `scale(${zoom})` }}>
          {PIPELINE.map((n, i) => (
            <div key={n.id} className="flex items-center gap-2">
              <button onClick={() => setSel(n.id)}
                className={cx("text-left rounded-lg border px-3 py-2.5 min-w-[150px] transition-colors focus-ring",
                  sel === n.id ? "border-blue bg-blue-wash" : "border-ink-700/70 bg-ink-850 hover:bg-ink-800",
                  n.internal && "border-dashed")}>
                <div className="flex items-center gap-1.5">
                  <span className="text-2xs font-mono text-ink-500">{String(i).padStart(2, "0")}</span>
                  {n.internal && <EyeOff className="h-3 w-3 text-violet" />}
                </div>
                <div className={cx("text-xs font-medium mt-1", sel === n.id ? "text-blue" : "text-ink-100")}>{n.label}</div>
                <div className="text-2xs font-mono text-ink-500 mt-0.5">{n.module.split("/").pop()}</div>
              </button>
              {i < PIPELINE.length - 1 && <ArrowRight className="h-3.5 w-3.5 text-ink-600 shrink-0" />}
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid lg:grid-cols-[1fr_320px] gap-4 items-start">
        <Panel title={node.label} subtitle={node.module}>
          {node.internal && (
            <div className="mb-4 rounded-md border border-violet/30 bg-violet-wash px-3 py-2 text-xs text-violet leading-relaxed">
              <strong>Env-internal.</strong> The defender never observes this stage. Attack
              success is decided here, from data flow — not from whether the defender was fooled.
              That separation is what stops a defender from "winning" by being confidently wrong.
            </div>
          )}
          <div className="space-y-4 text-xs">
            <Block k="Purpose" v={node.purpose} />
            <div className="grid sm:grid-cols-2 gap-4">
              <Block k="Input" v={node.input} mono />
              <Block k="Output" v={node.output} mono />
            </div>
            <Block k="Decision logic" v={node.processing} />
          </div>
          <div className="mt-4 pt-4 border-t hairline grid sm:grid-cols-2 gap-4 text-xs">
            <div>
              <div className="label mb-1.5">Upstream</div>
              {upstream.length ? upstream.map((u) => (
                <button key={u} onClick={() => setSel(u)} className="block text-blue hover:underline">
                  {PIPELINE.find((n) => n.id === u)!.label}
                </button>
              )) : <span className="text-ink-500">entry point</span>}
            </div>
            <div>
              <div className="label mb-1.5">Downstream</div>
              {downstream.length ? downstream.map((d) => (
                <button key={d} onClick={() => setSel(d)} className="block text-blue hover:underline">
                  {PIPELINE.find((n) => n.id === d)!.label}
                </button>
              )) : <span className="text-ink-500">terminal</span>}
            </div>
          </div>
        </Panel>

        <Panel title="Attack families" dense subtitle="Every one extended into a chained variant.">
          <div className="space-y-2">
            {FAMILIES.map((f) => (
              <div key={f.family} className="panel-tight px-2.5 py-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-medium text-ink-200">{f.label}</span>
                  <Badge tone="neutral" mono>≥{f.minChainLen}</Badge>
                </div>
                <p className="text-2xs text-ink-500 mt-1 leading-snug">{f.blurb}</p>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

const Block = ({ k, v, mono }: { k: string; v: string; mono?: boolean }) => (
  <div>
    <div className="label mb-1.5">{k}</div>
    <p className={cx("leading-relaxed text-ink-300", mono && "font-mono text-2xs")}>{v}</p>
  </div>
);
