export const DIVMOD_MATH_OPS: ReadonlySet<string> = new Set(['//', '%', 'tdiv', 'tmod']);
export const DIVISION_MATH_OPS: ReadonlySet<string> = new Set(['/', ...DIVMOD_MATH_OPS]);

function asInteger(value: number): number {
  return value === 0 ? 0 : value;
}

export function floorDiv(a: number, b: number): number {
  return asInteger(Math.floor(a / b));
}

export function floorMod(a: number, b: number): number {
  return asInteger(a - floorDiv(a, b) * b);
}

export function truncDiv(a: number, b: number): number {
  return asInteger(Math.trunc(a / b));
}

export function truncMod(a: number, b: number): number {
  return asInteger(a - truncDiv(a, b) * b);
}
