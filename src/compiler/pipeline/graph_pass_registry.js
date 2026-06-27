const _graphPasses = [];

export function registerGraphPass(factory, { phase = 'post', priority = 0 } = {}) {
  _graphPasses.push({ factory, phase, priority });
}

export function snapshotGraphPasses() {
  return [..._graphPasses];
}

export function graphPassesForPhase(phase, config, target, entries = _graphPasses) {
  return entries
    .filter((e) => e.phase === phase)
    .sort((a, b) => a.priority - b.priority)
    .map((e) => e.factory(config, target))
    .filter(Boolean);
}

export function clearGraphPasses() {
  _graphPasses.length = 0;
}
