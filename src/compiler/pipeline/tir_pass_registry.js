const _tirPasses = [];

export function registerTirPass(factory, { phase = 'post', priority = 0 } = {}) {
  _tirPasses.push({ factory, phase, priority });
}

export function snapshotTirPasses() {
  return [..._tirPasses];
}

export function tirPassesForPhase(phase, config, entries = _tirPasses) {
  return entries
    .filter((e) => e.phase === phase)
    .sort((a, b) => a.priority - b.priority)
    .map((e) => e.factory(config))
    .filter(Boolean);
}

export function clearTirPasses() {
  _tirPasses.length = 0;
}
