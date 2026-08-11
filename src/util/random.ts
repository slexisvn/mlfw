export type Rng = () => number;

const DEFAULT_SEED = 0x9e3779b9;

export function makeRng(seed?: number): Rng {
  let a = (seed ?? DEFAULT_SEED) >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let global: Rng | null = null;
let currentSeed: number | null = null;

export function manualSeed(seed: number): void {
  currentSeed = seed >>> 0;
  global = makeRng(currentSeed);
}

export function seed(): number | null {
  return currentSeed;
}

export function unseed(): void {
  global = null;
  currentSeed = null;
}

export function random(): number {
  return global ? global() : Math.random();
}

export function randomInt(n: number): number {
  return Math.floor(random() * n);
}
