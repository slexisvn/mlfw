export function gcd(a: number, b: number): number {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y !== 0) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x;
}

export function gcdOf(values: Iterable<number>): number {
  let g = 0;
  for (const value of values) {
    g = gcd(g, value);
    if (g === 1) return 1;
  }
  return g;
}
