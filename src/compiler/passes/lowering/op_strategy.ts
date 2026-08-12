export type ComputeFn = (...args: never[]) => unknown;
export type OpStrategyTarget = { kind?: string } | null | undefined;
export type ImplSpec = { name: string; compute: ComputeFn; plevel?: number; targetKind?: string | null };

export class OpImplementation {
  name: string;
  compute: ComputeFn;
  plevel: number;
  targetKind: string | null;

  constructor(name: string, compute: ComputeFn, plevel = 10, targetKind: string | null = null) {
    this.name = name;
    this.compute = compute;
    this.plevel = plevel;
    this.targetKind = targetKind;
  }
}

export class OpStrategy {
  opName: string;
  implementations: OpImplementation[];

  constructor(opName: string) {
    this.opName = opName;
    this.implementations = [];
  }

  addImplementation(name: string, compute: ComputeFn, plevel = 10, targetKind: string | null = null): this {
    const impl = new OpImplementation(name, compute, plevel, targetKind);
    const i = this.implementations.findIndex((e) => e.name === name);
    if (i >= 0) this.implementations[i] = impl;
    else this.implementations.push(impl);
    return this;
  }

  get candidates(): OpImplementation[] {
    return this.implementations;
  }

  best(): OpImplementation | null {
    let chosen: OpImplementation | null = null;
    for (const impl of this.implementations) {
      if (chosen === null || impl.plevel > chosen.plevel) chosen = impl;
    }
    return chosen;
  }
}

const strategyRegistry = new Map<string, OpStrategy>();
const viewCache = new Map<string, OpStrategy | null>();

export function registerOpStrategy(opName: string, { name, compute, plevel = 10, targetKind = null }: ImplSpec): OpStrategy {
  let strat = strategyRegistry.get(opName);
  if (!strat) {
    strat = new OpStrategy(opName);
    strategyRegistry.set(opName, strat);
  }
  strat.addImplementation(name, compute, plevel, targetKind);
  viewCache.clear();
  return strat;
}

export function unregisterOpStrategy(opName: string): boolean {
  const existed = strategyRegistry.delete(opName);
  viewCache.clear();
  return existed;
}

export function getOpStrategy(opName: string, target: OpStrategyTarget = null): OpStrategy | null {
  const full = strategyRegistry.get(opName);
  if (!full) return null;
  const kind = target ? target.kind ?? null : null;
  const cacheKey = `${opName}|${kind === null ? '' : kind}`;
  if (viewCache.has(cacheKey)) return viewCache.get(cacheKey) as OpStrategy | null;
  const view = new OpStrategy(opName);
  for (const impl of full.implementations) {
    if (impl.targetKind === null || impl.targetKind === kind) {
      view.implementations.push(impl);
    }
  }
  const result = view.implementations.length > 0 ? view : null;
  viewCache.set(cacheKey, result);
  return result;
}

export function selectImplementation(opName: string, target: OpStrategyTarget = null): OpImplementation | null {
  const strat = getOpStrategy(opName, target);
  return strat ? strat.best() : null;
}

export function hasOpStrategy(opName: string, target: OpStrategyTarget = null): boolean {
  return getOpStrategy(opName, target) !== null;
}
