const _rules = new Map();

export function registerVJPRule(opName, ruleFn) {
  _rules.set(opName, ruleFn);
}

export function getVJPRule(opName) {
  return _rules.get(opName) || null;
}

export function hasVJPRule(opName) {
  return _rules.has(opName);
}

export function listRegisteredOps() {
  return [..._rules.keys()];
}
