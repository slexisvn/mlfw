const _rules = new Map();
const _barriers = new Set();

export function registerVJPRule(opName, ruleFn) {
  _rules.set(opName, ruleFn);
}

export function getVJPRule(opName) {
  return _rules.get(opName) || null;
}

export function requireVJPRuleOrBarrier(opName) {
  const rule = _rules.get(opName);
  if (rule) return rule;
  if (_barriers.has(opName)) return null;
  throw new Error(`autodiff: op '${opName}' is on the gradient path but has no VJP rule and is not a registered gradient barrier. Register one with registerVJPRule('${opName}', ...) or registerGradientBarrier('${opName}').`);
}

export function findUnsupportedGradOps(ops) {
  const missing = new Set();
  for (const op of ops) {
    if (!_rules.has(op.opName) && !_barriers.has(op.opName)) missing.add(op.opName);
  }
  return [...missing];
}

export function hasVJPRule(opName) {
  return _rules.has(opName);
}

export function listRegisteredOps() {
  return [..._rules.keys()];
}

const _regionRules = new Map();

export function registerRegionVJP(opName, fn) {
  _regionRules.set(opName, fn);
}

export function getRegionVJP(opName) {
  return _regionRules.get(opName) || null;
}

export function registerGradientBarrier(opName) {
  _barriers.add(opName);
}

export function isGradientBarrier(opName) {
  return _barriers.has(opName);
}
