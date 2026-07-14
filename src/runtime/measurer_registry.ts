type Measurer = (...args: unknown[]) => unknown;

const _measurers = new Map<string, Measurer>();

export function registerMeasurer(kind: string, fn: Measurer): void {
  _measurers.set(kind, fn);
}

export function getMeasurer(kind: string): Measurer | null {
  return _measurers.get(kind) || null;
}

export function hasMeasurer(kind: string): boolean {
  return _measurers.has(kind);
}
