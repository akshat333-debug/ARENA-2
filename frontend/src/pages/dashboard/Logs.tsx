import { useEffect, useRef, useState } from "react";
import { Download } from "lucide-react";
import { useStore, LogLevel } from "../../store/useStore";
import { Button, EmptyState, Panel, SegMap } from "../../components/ui";
import { clock, cx } from "../../lib/format";

const LEVEL_STYLE: Record<LogLevel, { c: string; t: string }> = {
  info: { c: "text-ink-300", t: "INFO" },
  ok: { c: "text-green", t: " OK " },
  warn: { c: "text-amber", t: "WARN" },
  error: { c: "text-red-soft", t: "ERR " },
  debug: { c: "text-violet", t: "DBG " },
};

export default function Logs() {
  const s = useStore();
  const [level, setLevel] = useState<"all" | LogLevel>("all");
  const [follow, setFollow] = useState(true);
  const ref = useRef<HTMLDivElement>(null);

  const rows = s.logs.filter((l) => level === "all" || l.level === level);

  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [s.logs.length, follow]);

  const exportLogs = () => {
    const text = s.logs.map((l) => `${clock(l.t)}  ${LEVEL_STYLE[l.level].t}  ${l.stage.padEnd(9)}  ${l.msg}`).join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "arena-execution.log"; a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="h-full p-4 lg:p-5 flex flex-col min-h-0">
      <Panel
        className="flex-1 min-h-0"
        title="Execution monitor"
        subtitle="Generated as the pipeline runs. Nothing here is pre-written."
        actions={<>
          <SegMap value={level} onChange={setLevel} options={[
            { value: "all", label: "All" }, { value: "info", label: "Info" },
            { value: "warn", label: "Warn" }, { value: "error", label: "Error" },
            { value: "debug", label: "Debug" },
          ]} />
          <Button size="sm" variant={follow ? "primary" : "secondary"} onClick={() => setFollow(!follow)}>
            {follow ? "Following" : "Follow"}
          </Button>
          <Button size="sm" variant="secondary" onClick={exportLogs} disabled={!s.logs.length}>
            <Download className="h-3 w-3" />
          </Button>
        </>}
        bodyClass="p-0 flex flex-col">
        <div className="shrink-0 px-4 py-2 border-b hairline flex items-center gap-3 text-2xs text-ink-500">
          <span className="tnum">{rows.length} line(s)</span>
          {(["info", "ok", "warn", "error", "debug"] as LogLevel[]).map((lv) => {
            const n = s.logs.filter((l) => l.level === lv).length;
            return n ? <span key={lv} className={cx("tnum", LEVEL_STYLE[lv].c)}>{lv} {n}</span> : null;
          })}
        </div>

        {rows.length === 0 ? (
          <EmptyState title="No log output"
            body={level === "all" ? "Run the simulation to generate execution output." : `No ${level} entries — try a different level.`}
            action={level === "all"
              ? <Button size="sm" variant="primary" onClick={() => { s.loadPreset(s.presetId); s.run(); }}>Run demo</Button>
              : <Button size="sm" onClick={() => setLevel("all")}>Show all</Button>} />
        ) : (
          <div ref={ref} onScroll={(e) => {
              const el = e.currentTarget;
              if (el.scrollHeight - el.scrollTop - el.clientHeight > 40) setFollow(false);
            }}
            className="flex-1 min-h-[240px] overflow-y-auto bg-ink-950 font-mono text-2xs leading-relaxed p-3">
            {rows.map((l, i) => (
              <div key={i} className="flex gap-3 hover:bg-ink-900/60 px-1 rounded animate-fade-up">
                <span className="hidden sm:inline text-ink-600 shrink-0">{clock(l.t)}</span>
                <span className={cx("shrink-0 font-semibold", LEVEL_STYLE[l.level].c)}>{LEVEL_STYLE[l.level].t}</span>
                <span className="hidden sm:inline text-ink-500 shrink-0 w-16">{l.stage}</span>
                <span className="text-ink-200 break-all">{l.msg}</span>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
