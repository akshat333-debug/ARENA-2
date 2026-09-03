import { useMemo } from "react";
import { cx } from "../../lib/format";

/* Hand-rolled SVG. No chart library: full control over the technical look,
   zero runtime deps, and it renders identically offline. */

export function Sparkline({ data, tone = "#3DD8E8", height = 32, fill = true }: {
  data: number[]; tone?: string; height?: number; fill?: boolean;
}) {
  const d = useMemo(() => {
    if (data.length === 0) return null;
    const min = Math.min(...data), max = Math.max(...data);
    const span = max - min || 1;
    // One point still plots — as a flat mark at mid-height. Reporting "no data
    // yet" next to a panel that says "1 episode(s) this session" reads as a bug.
    const pts = data.map((v, i) => [
      data.length === 1 ? 50 : (i / (data.length - 1)) * 100,
      data.length === 1 ? 50 : 100 - ((v - min) / span) * 100,
    ]);
    return {
      single: data.length === 1,
      line: pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" "),
      area: `M0,100 L${pts.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" L")} L100,100 Z`,
    };
  }, [data]);
  if (!d) return <div style={{ height }} className="flex items-center text-2xs text-ink-500">no data yet</div>;
  const id = `sg-${tone.replace("#", "")}`;
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ height }} className="w-full overflow-visible">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={tone} stopOpacity=".28" />
          <stop offset="100%" stopColor={tone} stopOpacity="0" />
        </linearGradient>
      </defs>
      {fill && !d.single && <path d={d.area} fill={`url(#${id})`} />}
      {d.single
        // a zero-length path is not reliably stroked across browsers
        ? <circle cx="50" cy="50" r="2.5" fill={tone} vectorEffect="non-scaling-stroke" />
        : <path d={d.line} fill="none" stroke={tone} strokeWidth="1.6" vectorEffect="non-scaling-stroke"
            strokeLinejoin="round" strokeLinecap="round" />}
    </svg>
  );
}

export interface SeriesPoint { x: number; y: number; }
export interface Series { name: string; color: string; points: SeriesPoint[]; dashed?: boolean; band?: number[]; }

/** Line chart with optional ±band — used for the seed-variance figure, where
 *  the band IS the finding. */
export function LineChart({ series, height = 220, yLabel, xLabel, yMax, yMin = 0, xTicks }: {
  series: Series[]; height?: number; yLabel?: string; xLabel?: string;
  yMax?: number; yMin?: number; xTicks?: string[];
}) {
  const all = series.flatMap((s) => s.points);
  if (!all.length) return null;
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  const bandMax = Math.max(...series.flatMap((s) => s.band ? s.points.map((p, i) => p.y + (s.band![i] ?? 0)) : []), -Infinity);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = yMin, y1 = yMax ?? Math.max(Math.max(...ys), isFinite(bandMax) ? bandMax : 0) * 1.15;
  const P = { l: 44, r: 14, t: 12, b: 30 };
  const W = 640, H = height;
  const sx = (x: number) => P.l + ((x - x0) / (x1 - x0 || 1)) * (W - P.l - P.r);
  const sy = (y: number) => H - P.b - ((y - y0) / (y1 - y0 || 1)) * (H - P.t - P.b);
  const gridY = Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: height }}>
      {gridY.map((g, i) => (
        <g key={i}>
          <line x1={P.l} x2={W - P.r} y1={sy(g)} y2={sy(g)} stroke="#ffffff" strokeOpacity=".07" />
          <text x={P.l - 8} y={sy(g)} textAnchor="end" dominantBaseline="middle"
            className="fill-ink-400" style={{ fontSize: 10 }}>{g.toFixed(2)}</text>
        </g>
      ))}
      {series.map((s) => {
        const line = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ");
        const band = s.band && [
          ...s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y + s.band![i])}`),
          ...[...s.points].reverse().map((p, i) => `L${sx(p.x)},${sy(p.y - s.band![s.points.length - 1 - i])}`),
          "Z",
        ].join(" ");
        return (
          <g key={s.name}>
            {band && <path d={band} fill={s.color} fillOpacity=".13" />}
            <path d={line} fill="none" stroke={s.color} strokeWidth="2"
              strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" strokeLinecap="round" />
            {s.points.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="3" fill="#0C0E12" stroke={s.color} strokeWidth="2" />
            ))}
          </g>
        );
      })}
      {xTicks?.map((t, i) => (
        <text key={i} x={sx(x0 + ((x1 - x0) * i) / (xTicks.length - 1 || 1))} y={H - 10}
          textAnchor="middle" className="fill-ink-400" style={{ fontSize: 10 }}>{t}</text>
      ))}
      {yLabel && <text x={12} y={P.t + 4} className="fill-ink-500" style={{ fontSize: 10 }}>{yLabel}</text>}
      {xLabel && <text x={W - P.r} y={H - 10} textAnchor="end" className="fill-ink-500" style={{ fontSize: 10 }}>{xLabel}</text>}
    </svg>
  );
}

/** Horizontal bars with an error bar. The error bar is the point: a mean
 *  without its spread is how this project shipped two wrong headlines. */
export function BarsWithError({ rows, max = 1, height = 26, format = (v: number) => v.toFixed(3) }: {
  rows: { label: string; value: number; sd?: number; color: string; note?: string }[];
  max?: number; height?: number; format?: (v: number) => string;
}) {
  return (
    <div className="space-y-2.5">
      {rows.map((r) => {
        const w = Math.max(0, Math.min(1, r.value / max)) * 100;
        const lo = Math.max(0, (r.value - (r.sd ?? 0)) / max) * 100;
        const hi = Math.min(1, (r.value + (r.sd ?? 0)) / max) * 100;
        return (
          <div key={r.label}>
            <div className="flex items-baseline justify-between mb-1">
              <span className="text-xs text-ink-200">{r.label}</span>
              <span className="text-xs tnum text-ink-300">
                {format(r.value)}
                {r.sd !== undefined && <span className="text-ink-500"> ±{format(r.sd)}</span>}
              </span>
            </div>
            <div className="relative rounded bg-ink-850 overflow-visible" style={{ height }}>
              <div className="absolute inset-y-0 left-0 rounded transition-[width] duration-700"
                style={{ width: `${w}%`, background: r.color, opacity: .85 }} />
              {r.sd !== undefined && r.sd > 0 && (
                <div className="absolute top-1/2 -translate-y-1/2 h-2.5 border-x-2 border-white/50"
                  style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 0.5)}%` }}>
                  <div className="absolute top-1/2 -translate-y-1/2 w-full border-t border-white/50" />
                </div>
              )}
            </div>
            {r.note && <div className="mt-1 text-2xs text-ink-500">{r.note}</div>}
          </div>
        );
      })}
    </div>
  );
}

/** ROC curve — drawn from actual scored decisions, not a decorative arc. */
export function RocCurve({ curves, height = 240 }: {
  curves: { name: string; color: string; points: [number, number][]; auc: number }[];
  height?: number;
}) {
  const S = 240, P = 30;
  const sx = (x: number) => P + x * (S - P - 10);
  const sy = (y: number) => S - P - y * (S - P - 10);
  return (
    <div className="flex flex-col sm:flex-row gap-4 items-center">
      <svg viewBox={`0 0 ${S} ${S}`} style={{ maxHeight: height }} className="w-full max-w-[260px]">
        <rect x={P} y={10} width={S - P - 10} height={S - P - 10} fill="none" stroke="#ffffff" strokeOpacity=".08" />
        {[0.25, 0.5, 0.75].map((g) => (
          <g key={g}>
            <line x1={sx(g)} x2={sx(g)} y1={10} y2={sy(0)} stroke="#fff" strokeOpacity=".05" />
            <line x1={P} x2={sx(1)} y1={sy(g)} y2={sy(g)} stroke="#fff" strokeOpacity=".05" />
          </g>
        ))}
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#5A6273" strokeDasharray="4 4" strokeWidth="1" />
        {curves.map((c) => (
          <path key={c.name} d={c.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0])},${sy(p[1])}`).join(" ")}
            fill="none" stroke={c.color} strokeWidth="2" strokeLinejoin="round" />
        ))}
        <text x={sx(0.5)} y={S - 6} textAnchor="middle" className="fill-ink-400" style={{ fontSize: 9 }}>false-positive rate</text>
        <text x={10} y={sy(0.5)} textAnchor="middle" transform={`rotate(-90 10 ${sy(0.5)})`}
          className="fill-ink-400" style={{ fontSize: 9 }}>true-positive rate</text>
      </svg>
      <div className="space-y-1.5 w-full sm:w-auto">
        {curves.map((c) => (
          <div key={c.name} className="flex items-center gap-2 text-xs">
            <span className="h-0.5 w-4 rounded" style={{ background: c.color }} />
            <span className="text-ink-300 flex-1">{c.name}</span>
            <span className="tnum text-ink-200">AUC {c.auc.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Rank-based ROC, matching arena/eval/metrics.py (Mann-Whitney with mid-ranks). */
export function rocPoints(scores: number[], labels: number[]): { points: [number, number][]; auc: number } {
  const pairs = scores.map((s, i) => ({ s, y: labels[i] })).sort((a, b) => b.s - a.s);
  const P = labels.filter((l) => l === 1).length, N = labels.length - P;
  if (!P || !N) return { points: [[0, 0], [1, 1]], auc: 0.5 };
  const pts: [number, number][] = [[0, 0]];
  let tp = 0, fp = 0;
  for (const p of pairs) { p.y === 1 ? tp++ : fp++; pts.push([fp / N, tp / P]); }
  let auc = 0;
  for (let i = 1; i < pts.length; i++) auc += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2;
  return { points: pts, auc };
}

export function Gauge({ value, label, tone = "#3DD8E8", size = 92 }: {
  value: number; label?: string; tone?: string; size?: number;
}) {
  const r = 38, C = 2 * Math.PI * r, v = Math.max(0, Math.min(1, value));
  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg viewBox="0 0 100 100" className="-rotate-90 w-full h-full">
        <circle cx="50" cy="50" r={r} fill="none" stroke="#1E222B" strokeWidth="9" />
        <circle cx="50" cy="50" r={r} fill="none" stroke={tone} strokeWidth="9" strokeLinecap="round"
          strokeDasharray={C} strokeDashoffset={C * (1 - v)}
          style={{ transition: "stroke-dashoffset .7s cubic-bezier(.2,.8,.2,1)" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-semibold tnum leading-none">{(v * 100).toFixed(0)}<span className="text-xs text-ink-400">%</span></span>
        {label && <span className="text-2xs text-ink-400 mt-0.5">{label}</span>}
      </div>
    </div>
  );
}

export function StackedBar({ segments, height = 10 }: {
  segments: { value: number; color: string; label: string }[]; height?: number;
}) {
  const total = segments.reduce((a, s) => a + s.value, 0) || 1;
  return (
    <div className="flex rounded-full overflow-hidden" style={{ height }}>
      {segments.map((s, i) => (
        <div key={i} title={`${s.label}: ${s.value}`} className={cx("transition-all duration-500")}
          style={{ width: `${(s.value / total) * 100}%`, background: s.color }} />
      ))}
    </div>
  );
}
