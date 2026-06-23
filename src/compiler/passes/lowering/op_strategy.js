export class OpImplementation {
  constructor(name, compute, plevel = 10, targetKind = null) {
    this.name = name;
    this.compute = compute;
    this.plevel = plevel;
    this.targetKind = targetKind;
  }
}

export class OpStrategy {
  constructor(opName) {
    this.opName = opName;
    this.implementations = [];
  }

  addImplementation(name, compute, plevel = 10, targetKind = null) {
    const impl = new OpImplementation(name, compute, plevel, targetKind);
    const i = this.implementations.findIndex((e) => e.name === name);
    if (i >= 0) this.implementations[i] = impl;
    else this.implementations.push(impl);
    return this;
  }

  get candidates() {
    return this.implementations;
  }

  best() {
    let chosen = null;
    for (const impl of this.implementations) {
      if (chosen === null || impl.plevel > chosen.plevel) chosen = impl;
    }
    return chosen;
  }
}

const strategyRegistry = new Map();

export function registerOpStrategy(opName, { name, compute, plevel = 10, targetKind = null }) {
  let strat = strategyRegistry.get(opName);
  if (!strat) {
    strat = new OpStrategy(opName);
    strategyRegistry.set(opName, strat);
  }
  strat.addImplementation(name, compute, plevel, targetKind);
  return strat;
}

export function getOpStrategy(opName, target = null) {
  const full = strategyRegistry.get(opName);
  if (!full) return null;
  const kind = target ? target.kind : null;
  const view = new OpStrategy(opName);
  for (const impl of full.implementations) {
    if (impl.targetKind === null || impl.targetKind === kind) {
      view.implementations.push(impl);
    }
  }
  return view.implementations.length > 0 ? view : null;
}

export function selectImplementation(opName, target = null) {
  const strat = getOpStrategy(opName, target);
  return strat ? strat.best() : null;
}

export function hasOpStrategy(opName, target = null) {
  return getOpStrategy(opName, target) !== null;
}
