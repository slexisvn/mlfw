export function float64From(data: ArrayLike<number | bigint>): Float64Array {
  const out = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
  return out;
}

export function f64(data: ArrayLike<number | bigint>, index: number): number {
  return Number(data[index]);
}

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}
