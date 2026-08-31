import { Link } from "react-router-dom";
import { useState } from "react";
import {
  ArrowRight, Boxes, CircleCheck, CircleX, Cpu, GitBranch, Info, Layers,
  ShieldAlert, Swords, Target, Workflow as WorkflowIcon,
} from "lucide-react";
import { HeroFlow } from "../components/viz/HeroFlow";
import { Badge, Button, Panel } from "../components/ui";
import { BarsWithError } from "../components/viz";
import { DEFENDERS } from "../lib/policies";
import { FAMILIES } from "../lib/domain";
import { cx } from "../lib/format";

/** Provenance tag. Every number on this page carries one — the project spent
 *  two audits withdrawing claims that were really noise, so unlabelled figures
 *  are not something this page is willing to ship. */
function Src({ kind }: { kind: "measured" | "published" | "projected" | "illustrative" }) {
  const map = {
    measured: ["Measured", "green", "5-seed run in this repo, report/multiseed_baseline.json"],
    published: ["Published", "blue", "From the cited paper"],
    projected: ["Projected", "amber", "Prototype projection, not a measurement"],
    illustrative: ["Illustrative", "violet", "Illustrative scenario"],
  } as const;
  const [label, tone, title] = map[kind];
  return <Badge tone={tone as any} className="align-middle"><span title={title}>{label}</span></Badge>;
}

export default function Home() {
  return (
    <div className="min-h-full bg-ink-950">
      <Nav />
      <Hero />
      <Problem />
      <Different />
      <Outcomes />
      <UseCases />
      <FinalCta />
      <Footer />
    </div>
  );
}

function Nav() {
  return (
    <nav className="sticky top-0 z-30 h-14 border-b hairline bg-ink-950/85 backdrop-blur">
      <div className="mx-auto max-w-6xl h-full px-5 flex items-center gap-3">
        <div className="h-7 w-7 rounded-md bg-blue/15 border border-blue/30 grid place-items-center">
          <Swords className="h-3.5 w-3.5 text-blue" />
        </div>
        <span className="font-semibold tracking-tight">ARENA</span>
        <span className="hidden sm:inline text-2xs text-ink-500 border-l hairline pl-3 ml-1">
          Adversarial Tool-Use Security
        </span>
        <div className="ml-auto flex items-center gap-2">
          <a href="#how" className="hidden sm:inline text-xs text-ink-300 hover:text-ink-100 px-2">How it works</a>
          <Link to="/app"><Button size="sm" variant="primary">Launch Console <ArrowRight className="h-3.5 w-3.5" /></Button></Link>
        </div>
      </div>
    </nav>
  );
}

function Hero() {
  return (
    <header className="relative overflow-hidden border-b hairline">
      <div className="absolute inset-0 grid-bg opacity-40" />
      <div className="absolute -top-32 left-1/3 h-64 w-[36rem] rounded-full bg-blue/10 blur-3xl" />
      <div className="relative mx-auto max-w-6xl px-5 py-16 lg:py-20 grid lg:grid-cols-[1.05fr_1fr] gap-12 items-center">
        <div>
          <Badge tone="blue" className="mb-4">
            <Target className="h-3 w-3" /> Exploitability, not accuracy
          </Badge>
          <h1 className="text-4xl lg:text-[3.25rem] font-semibold tracking-tight leading-[1.05]">
            Your AI agent's defender<br />
            <span className="text-ink-400">has never met an attacker</span><br />
            that learns.
          </h1>
          <p className="mt-5 text-[15px] leading-relaxed text-ink-300 max-w-xl">
            ARENA is a benchmark where the attacker and the defender co-evolve. Instead of
            scoring a defender against a frozen list of known attacks, it trains a fresh
            attacker <em className="text-ink-200 not-italic">against your defender</em> and
            reports what still gets through.
          </p>
          <div className="mt-7 flex flex-wrap gap-2.5">
            <Link to="/app"><Button size="lg" variant="primary">Launch Interactive Console <ArrowRight className="h-4 w-4" /></Button></Link>
            <a href="#how"><Button size="lg" variant="secondary">See how it works</Button></a>
          </div>
          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-2xs text-ink-400">
            <span className="flex items-center gap-1.5"><Layers className="h-3.5 w-3.5" /> 6 attack families, all chained</span>
            <span className="flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5" /> Self-play PPO + opponent league</span>
            <span className="flex items-center gap-1.5"><GitBranch className="h-3.5 w-3.5" /> Runs fully offline</span>
          </div>
        </div>
        <HeroFlow />
      </div>
    </header>
  );
}

function Problem() {
  const steps = [
    { icon: Boxes, title: "Current reality", body: "MCP agents are handed real tools — credential stores, webhooks, payment rails. Guardrails inspect one call at a time.", tone: "neutral" as const },
    { icon: ShieldAlert, title: "The exposure", body: "TAMAS reports multi-agent frameworks failing 81–82% of the time against attacks that do not even adapt.", tone: "red" as const, src: "published" as const },
    { icon: CircleX, title: "The gap", body: "A chained attack is benign at every individual step. A per-call detector reaches 0.160 TPR at 5% FPR — it cannot see the chain.", tone: "amber" as const, src: "measured" as const },
    { icon: CircleCheck, title: "What ARENA adds", body: "Score the defender against an attacker trained to beat it. That number is exploitability, and it is the only one that survives contact with an adaptive adversary.", tone: "blue" as const },
  ];
  return (
    <section className="border-b hairline">
      <div className="mx-auto max-w-6xl px-5 py-16">
        <SectionHead eyebrow="The problem" title="Static evaluation flatters defenders"
          sub="Every step below is observable in the console. None of it is a mock-up." />
        <div className="grid md:grid-cols-4 gap-3">
          {steps.map((s, i) => (
            <div key={s.title} className="panel p-4 relative">
              <div className="flex items-center gap-2 mb-2.5">
                <s.icon className={cx("h-4 w-4",
                  s.tone === "red" ? "text-red" : s.tone === "amber" ? "text-amber" :
                  s.tone === "blue" ? "text-blue" : "text-ink-400")} />
                <span className="text-2xs font-mono text-ink-500">0{i + 1}</span>
              </div>
              <h3 className="text-sm font-medium mb-1.5">{s.title}</h3>
              <p className="text-xs leading-relaxed text-ink-400">{s.body}</p>
              {s.src && <div className="mt-2.5"><Src kind={s.src} /></div>}
            </div>
          ))}
        </div>

        <div className="mt-4 panel p-5">
          <div className="flex items-baseline justify-between mb-4 flex-wrap gap-2">
            <div>
              <h3 className="text-sm font-medium">Detection quality vs. the split chain</h3>
              <p className="text-xs text-ink-400 mt-0.5">TPR at a matched 5% false-positive budget. Higher is better.</p>
            </div>
            <Src kind="measured" />
          </div>
          <BarsWithError max={1}
            rows={DEFENDERS.map((d) => ({
              label: d.name, value: d.stats.tpr, sd: d.stats.tprSd,
              color: d.id === "arena_blue" ? "#3DD8E8" : d.id === "causal_monitor" ? "#9B8CFF" : "#5A6273",
              note: d.blurb,
            }))} />
          <p className="mt-4 text-2xs text-ink-500 flex items-start gap-1.5">
            <Info className="h-3 w-3 mt-px shrink-0" />
            Bars are means over 5 seeds; the white brackets are ±1 SD. Read the brackets
            before the bars — the co-evolved defender's spread is the widest here, and that
            variance is a finding in its own right.
          </p>
        </div>
      </div>
    </section>
  );
}

function Different() {
  const rows = [
    { t: "Attack model", a: "Fixed list of known exploits", b: "Attacker trained against your defender, from scratch, every evaluation" },
    { t: "Primary metric", a: "Accuracy / F1 on a frozen test set", b: "Exploitability — what a fresh best response still gets through" },
    { t: "Ground truth", a: "Did the detector fire?", b: "Did sensitive data actually cross the boundary? Decided by a taint graph the defender cannot see" },
    { t: "Attack shape", a: "Single malicious call", b: "Chained: no individual call is an attack, only the sequence is" },
    { t: "Failure mode caught", a: "Missed detections", b: "Also blanket paranoia — quarantining everything scores exactly as badly as allowing everything" },
  ];
  return (
    <section id="how" className="border-b hairline bg-ink-900/30">
      <div className="mx-auto max-w-6xl px-5 py-16">
        <SectionHead eyebrow="What's different" title="Five design decisions that change the result"
          sub="Not novelty for its own sake — each one closes a specific way a benchmark can lie to you." />
        <div className="panel overflow-hidden">
          <div className="grid grid-cols-[minmax(120px,1fr)_1.2fr_1.4fr] text-2xs label border-b hairline">
            <div className="px-4 py-2.5">Dimension</div>
            <div className="px-4 py-2.5 border-l hairline">Conventional benchmark</div>
            <div className="px-4 py-2.5 border-l hairline text-blue">ARENA</div>
          </div>
          {rows.map((r, i) => (
            <div key={r.t} className={cx("grid grid-cols-[minmax(120px,1fr)_1.2fr_1.4fr] text-xs",
              i % 2 === 1 && "bg-ink-850/40")}>
              <div className="px-4 py-3 font-medium text-ink-200">{r.t}</div>
              <div className="px-4 py-3 border-l hairline text-ink-400 flex gap-2">
                <CircleX className="h-3.5 w-3.5 text-ink-600 shrink-0 mt-px" />{r.a}
              </div>
              <div className="px-4 py-3 border-l hairline text-ink-200 flex gap-2">
                <CircleCheck className="h-3.5 w-3.5 text-blue shrink-0 mt-px" />{r.b}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Outcomes() {
  const cards = [
    { v: "0.783", l: "Allow-list exploitability", s: "measured" as const,
      d: "A per-call allow-list lets 78% of trained attacks through — independently reproducing TAMAS's ~80% figure." },
    { v: "6", l: "Attack families, all chained", s: "measured" as const,
      d: "Every TAMAS family extended so no attack completes in one call." },
    { v: "0.313", l: "Exploitability, co-evolved + causal features", s: "measured" as const,
      d: "vs 0.636 for the strongest static baseline — and it wins on all 5 of 5 seeds." },
    { v: "±0.274", l: "Seed spread of the plain co-evolved defender", s: "measured" as const,
      d: "An order of magnitude wider than any static baseline — a reliability problem single-seed evaluation cannot see." },
    { v: "0", l: "Network calls needed to demo", s: "measured" as const,
      d: "The console's simulation engine runs entirely in the browser." },
    { v: "20 pts", l: "TPR lost to LLM-planned attacks", s: "measured" as const,
      d: "The causal monitor drops 0.875 → 0.675 when a model plans the attack instead of our script." },
  ];
  return (
    <section className="border-b hairline">
      <div className="mx-auto max-w-6xl px-5 py-16">
        <SectionHead eyebrow="Outcomes" title="What the system actually produced"
          sub="Numbers from this repository's own runs. Nothing here is aspirational." />
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {cards.map((c) => (
            <div key={c.l} className="panel p-4">
              <div className="flex items-start justify-between gap-2">
                <span className="text-3xl font-semibold tnum tracking-tight">{c.v}</span>
                <Src kind={c.s} />
              </div>
              <div className="mt-1.5 text-sm font-medium text-ink-200">{c.l}</div>
              <p className="mt-1.5 text-xs leading-relaxed text-ink-400">{c.d}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function UseCases() {
  const [active, setActive] = useState(0);
  const cases = [
    { n: "01", title: "Pre-deployment review", who: "Security engineer shipping an MCP agent",
      normal: "Run the agent against a checklist of known prompt-injection strings. It passes. Ship it.",
      change: "ARENA trains a fresh attacker against your exact tool registry and reports what still lands.",
      result: "You learn the residual attack surface before an adversary does, not after.",
      family: "direct_prompt_injection" },
    { n: "02", title: "Guardrail bake-off", who: "Platform team choosing between vendors",
      normal: "Compare vendor-reported precision/recall on incomparable internal test sets.",
      change: "Every candidate is calibrated to the same 5% false-positive budget, then attacked by an equally-trained adversary.",
      result: "A single ranked table where the numbers mean the same thing.",
      family: "colluding_agents" },
    { n: "03", title: "Incident reconstruction", who: "Responder after a suspected tool-use breach",
      normal: "Read the raw call log and argue about which call was the problem.",
      change: "Replay the session; the taint graph marks exactly which calls carried the data and where it crossed the boundary.",
      result: "A defensible chain of custody from source to sink.",
      family: "byzantine_agent" },
    { n: "04", title: "Regression gate in CI", who: "Team shipping guardrail changes weekly",
      normal: "A rule change fixes one bypass and silently opens another.",
      change: "Exploitability is recomputed per change, over multiple seeds, with the spread reported.",
      result: "Guardrail regressions are caught as a number moving, not as an incident.",
      family: "indirect_prompt_injection" },
  ];
  const c = cases[active];
  return (
    <section className="border-b hairline bg-ink-900/30">
      <div className="mx-auto max-w-6xl px-5 py-16">
        <SectionHead eyebrow="Use cases" title="Where this fits in real work" />
        <div className="grid lg:grid-cols-[280px_1fr] gap-4">
          <div className="space-y-1.5">
            {cases.map((x, i) => (
              <button key={x.n} onClick={() => setActive(i)}
                className={cx("w-full text-left panel-tight px-3.5 py-3 transition-colors",
                  i === active ? "bg-ink-800 border-blue/40" : "hover:bg-ink-850")}>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-2xs text-ink-500">{x.n}</span>
                  <span className={cx("text-sm font-medium", i === active ? "text-blue" : "text-ink-200")}>{x.title}</span>
                </div>
                <div className="text-2xs text-ink-500 mt-0.5">{x.who}</div>
              </button>
            ))}
          </div>
          <Panel title={`Scenario ${c.n} — ${c.title}`} subtitle={c.who}
            actions={<Link to="/app/arena"><Button size="sm" variant="secondary">Open in console <ArrowRight className="h-3 w-3" /></Button></Link>}>
            <div className="grid sm:grid-cols-3 gap-3">
              {[["What normally happens", c.normal, "text-ink-400"],
                ["What ARENA changes", c.change, "text-ink-200"],
                ["Result", c.result, "text-blue"]].map(([h, b, tone]) => (
                <div key={h as string}>
                  <div className="label mb-1.5">{h as string}</div>
                  <p className={cx("text-xs leading-relaxed", tone as string)}>{b as string}</p>
                </div>
              ))}
            </div>
            <div className="mt-4 pt-4 border-t hairline">
              <div className="label mb-2">Attack family exercised</div>
              <div className="flex flex-wrap gap-1.5">
                {FAMILIES.map((f) => (
                  <Badge key={f.family} tone={f.family === c.family ? "red" : "neutral"}>
                    {f.label}
                  </Badge>
                ))}
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </section>
  );
}

function FinalCta() {
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-30" />
      <div className="absolute -bottom-24 left-1/2 -translate-x-1/2 h-56 w-[40rem] rounded-full bg-blue/10 blur-3xl" />
      <div className="relative mx-auto max-w-3xl px-5 py-20 text-center">
        <WorkflowIcon className="h-6 w-6 text-blue mx-auto mb-4" />
        <h2 className="text-3xl font-semibold tracking-tight">Enough explaining. Watch it run.</h2>
        <p className="mt-3 text-sm text-ink-400 max-w-lg mx-auto leading-relaxed">
          Play the attacker. Play the defender. Watch the pipeline execute call by call and
          see exactly where the chain breaks — entirely offline.
        </p>
        <Link to="/app" className="inline-block mt-7">
          <Button size="lg" variant="primary">Launch Interactive Console <ArrowRight className="h-4 w-4" /></Button>
        </Link>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t hairline">
      <div className="mx-auto max-w-6xl px-5 py-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-2xs text-ink-500">
        <span>ARENA — Adversarial Co-Evolving Benchmark for Agentic Tool-Use Security</span>
        <span className="ml-auto">Research prototype. Figures labelled by provenance.</span>
      </div>
    </footer>
  );
}

function SectionHead({ eyebrow, title, sub }: { eyebrow: string; title: string; sub?: string }) {
  return (
    <div className="mb-6 max-w-2xl">
      <div className="label text-blue mb-2">{eyebrow}</div>
      <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      {sub && <p className="mt-2 text-sm text-ink-400 leading-relaxed">{sub}</p>}
    </div>
  );
}
