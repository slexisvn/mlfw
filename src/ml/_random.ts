import { makeRng } from '../util/random.js';
import type { Rng } from './types.js';

export { makeRng };

export function randInt(rng: Rng, n: number): number {
  return Math.floor(rng() * n);
}

export function shuffledIndices(n: number, rng: Rng): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = randInt(rng, i + 1);
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx;
}
