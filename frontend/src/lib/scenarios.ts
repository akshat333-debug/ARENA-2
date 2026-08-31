/** Scenario generator — mirrors arena/scenarios.py.
 *  Family is decided BEFORE the registry is sampled, so the registry can be
 *  guaranteed to host a winning chain: no episode is unwinnable by construction.
 */
import {
  AttackFamily, AttackObjective, COMMON_TOOLS, DOMAINS, DOMAIN_TOOLS, DomainName,
  FAMILIES, Scenario, TASKS, ToolSpec, familySpec, isSink, isSource, isUntrustedRead, propagates,
} from "./domain";
import { Rng, makeRng } from "./rng";

export interface GenConfig {
  domains: DomainName[];
  families: AttackFamily[];
  nToolsMin: number;
  nToolsMax: number;
  adversarialRatio: number;
  maxSteps: number;
}

export const DEFAULT_GEN: GenConfig = {
  domains: DOMAINS, families: FAMILIES.map((f) => f.family),
  nToolsMin: 6, nToolsMax: 10, adversarialRatio: 0.5, maxSteps: 12,
};

function sampleRegistry(
  rng: Rng, domain: DomainName, n: number, needUntrustedRead: boolean,
): ToolSpec[] {
  const pool = [...COMMON_TOOLS, ...DOMAIN_TOOLS[domain]];
  const sources = pool.filter(isSource);
  const sinks = pool.filter(isSink);
  const props = pool.filter(propagates);
  // Guarantee a source -> propagator -> sink chain exists.
  const must: ToolSpec[] = [rng.pick(sources), rng.pick(props), rng.pick(sinks)];
  if (needUntrustedRead) must.push(rng.pick(pool.filter(isUntrustedRead)));
  const rest = rng.shuffle(pool.filter((t) => !must.some((m) => m.name === t.name)));
  const out = [...must, ...rest].slice(0, Math.max(n, must.length));
  return rng.shuffle(out);
}

let counter = 0;

export function sampleScenario(rng: Rng, cfg: GenConfig = DEFAULT_GEN, force?: {
  adversarial?: boolean; family?: AttackFamily;
}): Scenario {
  const domain = rng.pick(cfg.domains);
  const adversarial = force?.adversarial ?? rng.chance(cfg.adversarialRatio);
  const n = cfg.nToolsMin + rng.int(cfg.nToolsMax - cfg.nToolsMin + 1);
  const seed = rng.int(1e9);
  const id = `EP-${String(++counter).padStart(4, "0")}`;
  const task = rng.pick(TASKS[domain]);

  if (!adversarial) {
    return { id, domain, task, registry: sampleRegistry(rng, domain, n, false),
      adversarial: false, maxSteps: cfg.maxSteps, seed, objective: null };
  }

  const fam = force?.family ?? rng.pick(cfg.families);
  const spec = familySpec(fam);
  const registry = sampleRegistry(rng, domain, n, spec.requiresUntrustedRead);

  const sourcePool = spec.requiresUntrustedRead
    ? registry.filter(isUntrustedRead)
    : registry.filter(isSource);
  const sinkPool = registry.filter(isSink);

  const objective: AttackObjective = {
    kind: spec.kind, family: fam,
    sourceTools: [rng.pick(sourcePool).name],
    sinkTools: sinkPool.map((t) => t.name),
    minChainLen: spec.minChainLen,
    requiresUntrustedRead: spec.requiresUntrustedRead,
  };
  return { id, domain, task, registry, adversarial: true, maxSteps: cfg.maxSteps, seed, objective };
}

/** The minimal winning chain a scripted attacker would play. */
export function winningChain(sc: Scenario): ToolSpec[] {
  if (!sc.objective) return [];
  const obj = sc.objective;
  const src = sc.registry.find((t) => t.name === obj.sourceTools[0])!;
  const prop = sc.registry.find(propagates)!;
  const sink = obj.kind === "injected_control"
    ? sc.registry.find((t) => t.sideEffect === "exec") ?? sc.registry.find(isSink)!
    : sc.registry.find((t) => obj.sinkTools.includes(t.name))!;
  const nProp = Math.max(1, obj.minChainLen - 2);
  return [src, ...Array(nProp).fill(prop), sink];
}

export function resetScenarioCounter() { counter = 0; }
export { makeRng };
