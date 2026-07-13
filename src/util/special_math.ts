export const LANCZOS_G = 7;
export const LANCZOS_COEFFS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export const ERF_A = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
export const ERF_P = 0.3275911;

export const DIGAMMA_SHIFT = 6;
export const DIGAMMA_SERIES = [-1 / 12, 1 / 120, -1 / 252, 1 / 240, -1 / 132];

export function erfScalar(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  const ax = Math.abs(x);
  const t = 1 / (1 + ERF_P * ax);
  let poly = 0;
  for (let i = ERF_A.length - 1; i >= 0; i--) poly = poly * t + ERF_A[i];
  return sign * (1 - poly * t * Math.exp(-ax * ax));
}

export function erfcScalar(x: number): number {
  return 1 - erfScalar(x);
}

export function lgammaScalar(x: number): number {
  if (x < 0.5) {
    return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgammaScalar(1 - x);
  }
  const z = x - 1;
  let sum = LANCZOS_COEFFS[0];
  for (let i = 1; i < LANCZOS_COEFFS.length; i++) sum += LANCZOS_COEFFS[i] / (z + i);
  const t = z + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

export function gammaScalar(x: number): number {
  if (x < 0.5) {
    return Math.PI / (Math.sin(Math.PI * x) * gammaScalar(1 - x));
  }
  return Math.exp(lgammaScalar(x));
}

export function digammaScalar(x: number): number {
  let result = 0;
  let z = x;
  while (z < DIGAMMA_SHIFT) {
    result -= 1 / z;
    z += 1;
  }
  const inv = 1 / z;
  const inv2 = inv * inv;
  result += Math.log(z) - 0.5 * inv;
  let term = inv2;
  for (const c of DIGAMMA_SERIES) {
    result += c * term;
    term *= inv2;
  }
  return result;
}
