import { ReactNode, forwardRef } from "react";
import { cx } from "../../lib/format";

/* ---------------------------------------------------------------- Button */
type BtnVariant = "primary" | "secondary" | "ghost" | "danger" | "success";
type BtnSize = "sm" | "md" | "lg";

const BTN: Record<BtnVariant, string> = {
  primary: "bg-blue text-ink-950 hover:bg-blue-soft font-semibold",
  secondary: "bg-ink-800 text-ink-100 hover:bg-ink-700 border border-ink-600/80",
  ghost: "text-ink-300 hover:text-ink-100 hover:bg-ink-800/70",
  danger: "bg-red text-white hover:bg-red-soft font-semibold",
  success: "bg-green text-ink-950 hover:brightness-110 font-semibold",
};
const SIZE: Record<BtnSize, string> = {
  sm: "h-7 px-2.5 text-xs gap-1.5 rounded-md",
  md: "h-9 px-3.5 text-sm gap-2 rounded-lg",
  lg: "h-11 px-5 text-[15px] gap-2 rounded-lg",
};

export const Button = forwardRef<HTMLButtonElement, {
  variant?: BtnVariant; size?: BtnSize; children: ReactNode;
} & React.ButtonHTMLAttributes<HTMLButtonElement>>(
  ({ variant = "secondary", size = "md", className, children, ...p }, ref) => (
    <button ref={ref} {...p}
      className={cx("inline-flex items-center justify-center transition-colors focus-ring select-none",
        "disabled:opacity-40 disabled:pointer-events-none", BTN[variant], SIZE[size], className)}>
      {children}
    </button>
  ));
Button.displayName = "Button";

/* ----------------------------------------------------------------- Badge */
type Tone = "neutral" | "blue" | "red" | "amber" | "green" | "violet";
const TONE: Record<Tone, string> = {
  neutral: "bg-ink-800 text-ink-300 border-ink-600/70",
  blue: "bg-blue-wash text-blue border-blue/30",
  red: "bg-red-wash text-red-soft border-red/30",
  amber: "bg-amber-wash text-amber border-amber/30",
  green: "bg-green-wash text-green border-green/30",
  violet: "bg-violet-wash text-violet border-violet/30",
};
export function Badge({ tone = "neutral", children, className, mono }: {
  tone?: Tone; children: ReactNode; className?: string; mono?: boolean;
}) {
  return (
    <span className={cx("inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-2xs font-medium",
      mono && "font-mono", TONE[tone], className)}>{children}</span>
  );
}

/* ------------------------------------------------------------------ Dot */
export function Dot({ tone = "neutral", pulse }: { tone?: Tone; pulse?: boolean }) {
  const c: Record<Tone, string> = {
    neutral: "bg-ink-400", blue: "bg-blue", red: "bg-red",
    amber: "bg-amber", green: "bg-green", violet: "bg-violet",
  };
  return (
    <span className="relative inline-flex h-1.5 w-1.5">
      {pulse && <span className={cx("absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping", c[tone])} />}
      <span className={cx("relative inline-flex h-1.5 w-1.5 rounded-full", c[tone])} />
    </span>
  );
}

/* ---------------------------------------------------------------- Panel */
export function Panel({ title, subtitle, actions, children, className, bodyClass, dense }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode;
  children: ReactNode; className?: string; bodyClass?: string; dense?: boolean;
}) {
  return (
    <section className={cx("panel flex flex-col min-h-0", className)}>
      {(title || actions) && (
        <header className={cx("flex items-start justify-between gap-3 border-b hairline",
          dense ? "px-3 py-2" : "px-4 py-3")}>
          <div className="min-w-0">
            {title && <h3 className="text-sm font-semibold text-ink-100 leading-tight">{title}</h3>}
            {subtitle && <p className="text-xs text-ink-400 mt-0.5 leading-snug">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-1.5 shrink-0">{actions}</div>}
        </header>
      )}
      <div className={cx("min-h-0 flex-1", bodyClass ?? (dense ? "p-3" : "p-4"))}>{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------------ Stat */
export function Stat({ label, value, unit, delta, tone = "neutral", hint, sub }: {
  label: string; value: ReactNode; unit?: string; delta?: string;
  tone?: Tone; hint?: string; sub?: ReactNode;
}) {
  const vc: Record<Tone, string> = {
    neutral: "text-ink-100", blue: "text-blue", red: "text-red-soft",
    amber: "text-amber", green: "text-green", violet: "text-violet",
  };
  return (
    <div className="panel-tight px-3.5 py-3" title={hint}>
      <div className="label">{label}</div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span className={cx("text-2xl font-semibold tnum leading-none", vc[tone])}>{value}</span>
        {unit && <span className="text-xs text-ink-400">{unit}</span>}
        {delta && <span className="ml-auto text-2xs text-ink-400 tnum">{delta}</span>}
      </div>
      {sub && <div className="mt-1.5 text-2xs text-ink-400">{sub}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- Toggle */
export function SegMap<T extends string | number>({ value, options, onChange, size = "sm" }: {
  value: T; options: { value: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void; size?: "sm" | "md";
}) {
  return (
    <div className="inline-flex rounded-lg bg-ink-850 border border-ink-700/70 p-0.5">
      {options.map((o) => (
        <button key={String(o.value)} title={o.title} onClick={() => onChange(o.value)}
          className={cx("rounded-md transition-colors focus-ring",
            size === "sm" ? "px-2 h-6 text-2xs" : "px-2.5 h-7 text-xs",
            o.value === value ? "bg-ink-700 text-ink-100 font-medium" : "text-ink-400 hover:text-ink-200")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* --------------------------------------------------------------- Drawer */
export function Drawer({ open, onClose, title, subtitle, children, width = "max-w-xl" }: {
  open: boolean; onClose: () => void; title: ReactNode;
  subtitle?: ReactNode; children: ReactNode; width?: string;
}) {
  return (
    <div className={cx("fixed inset-0 z-50", !open && "pointer-events-none")} aria-hidden={!open}>
      <div onClick={onClose}
        className={cx("absolute inset-0 bg-ink-950/70 backdrop-blur-[2px] transition-opacity",
          open ? "opacity-100" : "opacity-0")} />
      <aside className={cx("absolute right-0 top-0 h-full w-full bg-ink-900 border-l hairline",
        "shadow-2xl transition-transform duration-300 ease-out flex flex-col", width,
        open ? "translate-x-0" : "translate-x-full")}>
        <header className="flex items-start justify-between gap-3 px-5 py-4 border-b hairline">
          <div className="min-w-0">
            <h3 className="text-base font-semibold">{title}</h3>
            {subtitle && <p className="text-xs text-ink-400 mt-0.5">{subtitle}</p>}
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>Close</Button>
        </header>
        <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
      </aside>
    </div>
  );
}

/* ------------------------------------------------------------- EmptyState */
export function EmptyState({ icon, title, body, action }: {
  icon?: ReactNode; title: string; body?: string; action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-12 px-6">
      {icon && <div className="mb-3 text-ink-500">{icon}</div>}
      <p className="text-sm font-medium text-ink-200">{title}</p>
      {body && <p className="text-xs text-ink-400 mt-1 max-w-sm leading-relaxed">{body}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ Bar */
export function Meter({ value, tone = "blue", height = "h-1.5" }: {
  value: number; tone?: Tone; height?: string;
}) {
  const c: Record<Tone, string> = {
    neutral: "bg-ink-400", blue: "bg-blue", red: "bg-red",
    amber: "bg-amber", green: "bg-green", violet: "bg-violet",
  };
  return (
    <div className={cx("w-full rounded-full bg-ink-800 overflow-hidden", height)}>
      <div className={cx("h-full rounded-full transition-[width] duration-500 ease-out", c[tone])}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

/* --------------------------------------------------------------- Tooltip */
export function Hint({ children, text }: { children: ReactNode; text: string }) {
  return (
    <span className="group relative inline-flex">
      {children}
      <span className="pointer-events-none absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-40
        whitespace-nowrap rounded-md bg-ink-800 border hairline px-2 py-1 text-2xs text-ink-200
        opacity-0 group-hover:opacity-100 transition-opacity shadow-xl">{text}</span>
    </span>
  );
}
