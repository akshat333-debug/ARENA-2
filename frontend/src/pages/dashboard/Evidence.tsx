import { useMemo, useState } from "react";
import { Download, Search, ShieldAlert, SlidersHorizontal } from "lucide-react";
import { useStore } from "../../store/useStore";
import { Badge, Button, Drawer, EmptyState, Panel, SegMap } from "../../components/ui";
import { ALLOW, FLAG, QUARANTINE, isPrivileged, isSink, isSource, isUntrustedRead, propagates } from "../../lib/domain";
import { cx, num } from "../../lib/format";

type Filter = "all" | "onpath" | "tainted" | "flagged";

export default function Evidence() {
  const s = useStore();
  const ep = s.episode;
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [sel, setSel] = useState<number | null>(null);
  const [sortKey, setSortKey] = useState<"step" | "tool" | "integrity">("step");

  const rows = useMemo(() => {
    if (!ep) return [];
    const tagsAt = (step: number) =>
      ep.tracker.artifacts.filter((a) => a.step === step).flatMap((a) => a.tags);
    let r = ep.history.map((h) => {
      const tags = tagsAt(h.stepIndex);
      const sensitive = tags.some((t) => t.color === "sensitive");
      const untrusted = tags.some((t) => t.color === "untrusted");
      return {
        id: `${ep.scenario.id}-C${String(h.stepIndex).padStart(2, "0")}`,
        step: h.stepIndex, tool: h.tool, verdict: h.verdict,
        onPath: h.onAttackPath, sensitive, untrusted, tags,
        integrity: h.verdict === QUARANTINE ? 0 : sensitive ? 0.35 : untrusted ? 0.6 : 1,
        confidence: sensitive ? 0.92 : untrusted ? 0.78 : 0.4,
      };
    });
    if (filter === "onpath") r = r.filter((x) => x.onPath);
    if (filter === "tainted") r = r.filter((x) => x.sensitive || x.untrusted);
    if (filter === "flagged") r = r.filter((x) => x.verdict !== ALLOW);
    if (q.trim()) {
      const k = q.toLowerCase();
      r = r.filter((x) => x.tool.name.includes(k) || x.id.toLowerCase().includes(k) || x.tool.sideEffect.includes(k));
    }
    r.sort((a, b) => sortKey === "step" ? a.step - b.step
      : sortKey === "tool" ? a.tool.name.localeCompare(b.tool.name) : a.integrity - b.integrity);
    return r;
  }, [ep, ep?.history.length, ep?.ended, q, filter, sortKey]);

  const item = sel !== null ? rows.find((r) => r.step === sel) : null;

  const exportJson = () => {
    const blob = new Blob([JSON.stringify({ scenario: ep?.scenario.id, rows: rows.map((r) => ({
      id: r.id, step: r.step, tool: r.tool.name, sideEffect: r.tool.sideEffect,
      verdict: r.verdict, onAttackPath: r.onPath, integrity: r.integrity,
    })) }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${ep?.scenario.id ?? "arena"}-evidence.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    s.log("ok", "export", `Exported ${rows.length} evidence rows.`);
  };

  if (!ep) return <div className="p-6"><EmptyState title="No episode loaded" /></div>;

  return (
    <div className="p-4 lg:p-5 space-y-4">
      <Panel
        title="Evidence ledger"
        subtitle={`Every executed call in ${ep.scenario.id}, with its taint state and integrity.`}
        actions={<>
          <SegMap value={filter} onChange={setFilter} options={[
            { value: "all", label: "All" }, { value: "onpath", label: "On path" },
            { value: "tainted", label: "Tainted" }, { value: "flagged", label: "Flagged" },
          ]} />
          <Button size="sm" variant="secondary" onClick={exportJson}><Download className="h-3 w-3" />Export</Button>
        </>}
        bodyClass="p-0">
        <div className="px-4 py-2.5 border-b hairline flex items-center gap-2">
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-ink-500" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search tool, id, side effect…"
              className="w-full bg-ink-850 border border-ink-700/70 rounded-md pl-8 pr-2 h-8 text-xs focus-ring placeholder:text-ink-600" />
          </div>
          <span className="text-2xs text-ink-500 tnum">{rows.length} row(s)</span>
          <div className="ml-auto flex items-center gap-1.5 text-2xs text-ink-500">
            <SlidersHorizontal className="h-3 w-3" />
            <SegMap value={sortKey} onChange={setSortKey} options={[
              { value: "step", label: "Step" }, { value: "tool", label: "Tool" }, { value: "integrity", label: "Integrity" },
            ]} />
          </div>
        </div>

        {rows.length === 0 ? (
          <EmptyState title={q || filter !== "all" ? "No matching evidence" : "No calls executed yet"}
            body={q || filter !== "all" ? "Adjust the filter or clear the search." : "Run the simulation to populate the ledger."}
            action={q || filter !== "all"
              ? <Button size="sm" onClick={() => { setQ(""); setFilter("all"); }}>Clear filters</Button>
              : <Button size="sm" variant="primary" onClick={() => { s.run(); }}>Run</Button>} />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="label border-b hairline">
                  {["Evidence ID", "Step", "Tool", "Side effect", "Verdict", "Taint", "Integrity", "Conf.", ""].map((h) => (
                    <th key={h} className="text-left font-medium px-3 py-2 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-800">
                {rows.map((r) => (
                  <tr key={r.id} onClick={() => setSel(r.step)}
                    className={cx("cursor-pointer transition-colors hover:bg-ink-850",
                      r.onPath && ep.ended && "bg-red-wash/30")}>
                    <td className="px-3 py-2 font-mono text-2xs text-ink-400 whitespace-nowrap">{r.id}</td>
                    <td className="px-3 py-2 tnum text-ink-300">{r.step}</td>
                    <td className="px-3 py-2 font-mono text-ink-100 whitespace-nowrap">{r.tool.name}</td>
                    <td className="px-3 py-2 text-ink-400 whitespace-nowrap">{r.tool.sideEffect}</td>
                    <td className="px-3 py-2">
                      <Badge tone={r.verdict === QUARANTINE ? "red" : r.verdict === FLAG ? "amber" : "neutral"}>
                        {r.verdict === QUARANTINE ? "quarantine" : r.verdict === FLAG ? "flag" : "allow"}
                      </Badge>
                    </td>
                    <td className="px-3 py-2">
                      {r.sensitive ? <Badge tone="red">sensitive</Badge>
                        : r.untrusted ? <Badge tone="violet">untrusted</Badge>
                        : <span className="text-ink-600">—</span>}
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-1.5">
                        <div className="w-12 h-1 rounded-full bg-ink-800 overflow-hidden">
                          <div className="h-full rounded-full" style={{
                            width: `${r.integrity * 100}%`,
                            background: r.integrity > 0.8 ? "#3ED598" : r.integrity > 0.5 ? "#F2B33D" : "#F04A5E",
                          }} />
                        </div>
                        <span className="tnum text-2xs text-ink-400">{num(r.integrity, 2)}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2 tnum text-ink-400">{num(r.confidence, 2)}</td>
                    <td className="px-3 py-2 text-right">
                      <span className="text-2xs text-blue">Inspect</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Drawer open={item !== null} onClose={() => setSel(null)}
        title={item?.id ?? ""} subtitle={item ? `${item.tool.name} · step ${item.step}` : ""}>
        {item && (
          <div className="space-y-4 text-xs">
            <div className="panel-tight p-3">
              <div className="label mb-1.5">Declared metadata (observable)</div>
              <dl className="grid grid-cols-2 gap-y-1.5">
                <Dt k="Side effect" v={item.tool.sideEffect} />
                <Dt k="Category" v={item.tool.category} />
                <Dt k="Sensitivity" v={String(item.tool.sensitivity)} />
                <Dt k="Privileged" v={isPrivileged(item.tool) ? "yes" : "no"} />
              </dl>
              <p className="mt-2 text-ink-400 leading-relaxed">{item.tool.description}</p>
            </div>

            <div className="panel-tight p-3 border-violet/30">
              <div className="label mb-1.5 text-violet">Taint state (env-internal ground truth)</div>
              {item.tags.length === 0
                ? <p className="text-ink-500">This call introduced no taint.</p>
                : <div className="space-y-1.5">
                  {item.tags.map((t, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <Badge tone={t.color === "sensitive" ? "red" : "violet"}>{t.color}</Badge>
                      <span className="font-mono text-2xs text-ink-300">from {t.originTool} @ step {t.originStep}</span>
                    </div>
                  ))}
                </div>}
              <div className="mt-2.5 pt-2.5 border-t hairline flex flex-wrap gap-1.5">
                {isSource(item.tool) && <Badge tone="red">taint source</Badge>}
                {propagates(item.tool) && <Badge tone="violet">propagator</Badge>}
                {isSink(item.tool) && <Badge tone="amber">boundary sink</Badge>}
                {isUntrustedRead(item.tool) && <Badge tone="violet">untrusted read</Badge>}
                {ep.ended && item.onPath && <Badge tone="red"><ShieldAlert className="h-3 w-3" />on attack path</Badge>}
              </div>
            </div>

            <div className="panel-tight p-3">
              <div className="label mb-1.5">Adjudication</div>
              <dl className="grid grid-cols-2 gap-y-1.5">
                <Dt k="Verdict" v={item.verdict === QUARANTINE ? "quarantine" : item.verdict === FLAG ? "flag" : "allow"} />
                <Dt k="Integrity" v={num(item.integrity, 2)} />
                <Dt k="Confidence" v={num(item.confidence, 2)} />
                <Dt k="Chain position" v={`${item.step + 1} of ${ep.history.length}`} />
              </dl>
            </div>
          </div>
        )}
      </Drawer>
    </div>
  );
}

const Dt = ({ k, v }: { k: string; v: string }) => (<>
  <dt className="text-ink-500">{k}</dt>
  <dd className="text-ink-200 font-mono text-2xs text-right">{v}</dd>
</>);
