import { useEffect, useState } from "react";
import { cx } from "../../lib/format";

/** The hero visual: a live split-chain exfiltration replaying against two
 *  defenders. It is the product's thesis in one picture — the per-call
 *  defender allows every step because no single call is an attack; the
 *  trajectory defender stops it at the sink. */
const CHAIN = [
  { tool: "read_env_file", se: "read_sensitive", tainted: true },
  { tool: "summarize", se: "transform", tainted: true },
  { tool: "format_report", se: "transform", tainted: true },
  { tool: "post_webhook", se: "external_sink", tainted: true },
];

export function HeroFlow() {
  const [step, setStep] = useState(-1);
  useEffect(() => {
    const id = setInterval(() => setStep((s) => (s >= CHAIN.length + 1 ? -1 : s + 1)), 1050);
    return () => clearInterval(id);
  }, []);

  const perCallVerdict = (i: number) => (i <= step ? "allow" : null);
  const trajVerdict = (i: number) =>
    i > step ? null : i === CHAIN.length - 1 ? "quarantine" : i >= 1 ? "flag" : "allow";

  return (
    <div className="panel p-5 grid-bg relative overflow-hidden">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-blue/50 to-transparent" />
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="label">Live · split-chain exfiltration</div>
          <div className="text-xs text-ink-300 mt-1">
            No single call is an attack. The <span className="text-red-soft">sequence</span> is.
          </div>
        </div>
        <span className="text-2xs font-mono text-ink-500">finance · EP-0001</span>
      </div>

      <div className="space-y-1.5">
        {CHAIN.map((c, i) => {
          const on = i <= step;
          return (
            <div key={c.tool}
              className={cx("relative grid grid-cols-[1fr_auto_auto] items-center gap-2 rounded-lg border px-3 py-2 transition-all duration-500",
                on ? "border-ink-600 bg-ink-850" : "border-ink-800 bg-ink-900/40 opacity-45")}>
              {on && c.tainted && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 h-6 w-0.5 rounded-full bg-red" />
              )}
              <div className="min-w-0">
                <div className="font-mono text-xs text-ink-100 truncate">{c.tool}</div>
                <div className="text-2xs text-ink-500">{c.se}</div>
              </div>
              <VerdictChip v={perCallVerdict(i)} muted />
              <VerdictChip v={trajVerdict(i)} />
            </div>
          );
        })}
      </div>

      <div className="mt-3 grid grid-cols-[1fr_auto_auto] gap-2 text-2xs">
        <span />
        <span className="w-[86px] text-center text-ink-500">per-call</span>
        <span className="w-[86px] text-center text-ink-500">trajectory</span>
      </div>

      <div className={cx("mt-4 rounded-lg border px-3 py-2.5 text-xs transition-all duration-500",
        step >= CHAIN.length
          ? "border-green/30 bg-green-wash text-green"
          : "border-ink-700 bg-ink-850 text-ink-400")}>
        {step >= CHAIN.length
          ? "Trajectory defender quarantined at the sink — credentials never left the session."
          : "Replaying…"}
      </div>
      <div className={cx("mt-1.5 rounded-lg border px-3 py-2.5 text-xs transition-all duration-500",
        step >= CHAIN.length ? "border-red/30 bg-red-wash text-red-soft" : "border-ink-700 bg-ink-850 text-ink-400")}>
        {step >= CHAIN.length
          ? "Per-call defender allowed all four — it never saw a sequence."
          : "…"}
      </div>
    </div>
  );
}

function VerdictChip({ v, muted }: { v: string | null; muted?: boolean }) {
  const tone = v === "quarantine" ? "border-green/40 bg-green-wash text-green"
    : v === "flag" ? "border-amber/40 bg-amber-wash text-amber"
    : v === "allow" ? (muted ? "border-red/30 bg-red-wash text-red-soft" : "border-ink-600 bg-ink-800 text-ink-300")
    : "border-ink-800 bg-transparent text-transparent";
  return (
    <span className={cx("w-[86px] text-center rounded border px-1.5 py-1 text-2xs font-mono transition-all duration-300", tone)}>
      {v ?? "—"}
    </span>
  );
}
