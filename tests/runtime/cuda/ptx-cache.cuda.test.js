import { describe, it, expect, beforeAll } from 'vitest';
import { mkdtempSync, rmSync, existsSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const cacheDir = mkdtempSync(join(tmpdir(), 'mlfw-ptx-'));
process.env.MLFW_PTX_CACHE_DIR = cacheDir;

const KERNEL = 'k_ptx_cache_probe';
const SOURCE = `__global__ void ${KERNEL}(float* out, const float* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * 3.0f + 1.0f;
}`;

let program, ptxCache;

beforeAll(async () => {
  program = await import('../../../src/runtime/node/cuda/program.js');
  ptxCache = await import('../../../src/runtime/node/cuda/ptx_cache.js');
});

function countEntries() {
  if (!existsSync(cacheDir)) return 0;
  let n = 0;
  for (const bucket of readdirSync(cacheDir)) {
    for (const f of readdirSync(join(cacheDir, bucket))) if (f.endsWith('.ptx')) n++;
  }
  return n;
}

describe('PTX disk cache', () => {
  it('writes a cache entry on the first compile of a source', () => {
    expect(countEntries()).toBe(0);
    const ptx = program.compileToPTX(SOURCE, KERNEL);
    expect(ptx.length).toBeGreaterThan(0);
    expect(countEntries()).toBe(1);
    expect(ptxCache.ptxCacheStats().writes).toBeGreaterThan(0);
  });

  it('serves a byte-identical PTX from disk when the in-memory cache is cold', () => {
    const first = program.compileToPTX(SOURCE, KERNEL);
    ptxCache.resetPtxCacheStats();
    program.clearProgramCache();

    const second = program.compileToPTX(SOURCE, KERNEL);
    const stats = ptxCache.ptxCacheStats();

    expect(stats.hits).toBe(1);
    expect(stats.writes).toBe(0);
    expect(Array.from(second)).toEqual(Array.from(first));
  });

  it('a cold-cache hit is far cheaper than an NVRTC compile', () => {
    const name = KERNEL + '_timed';
    const body = Array.from({ length: 64 }, (_, k) => `  acc = fmaf(acc, a[(i + ${k}) % n], ${k}.5f);`).join('\n');
    const uniq = `__global__ void ${name}(float* out, const float* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = a[i];
${body}
  out[i] = acc;
}`;

    program.clearProgramCache();
    let t = performance.now();
    program.compileToPTX(uniq, name);
    const compileMs = performance.now() - t;

    const REPEATS = 5;
    program.clearProgramCache();
    program.compileToPTX(uniq, name);
    t = performance.now();
    for (let k = 0; k < REPEATS; k++) {
      program.clearProgramCache();
      program.compileToPTX(uniq, name);
    }
    const diskMs = (performance.now() - t) / REPEATS;

    expect(compileMs).toBeGreaterThan(20);
    expect(diskMs).toBeLessThan(compileMs / 5);
  });

  it('keys the entry on the kernel source, not just the name', () => {
    const before = countEntries();
    program.compileToPTX(SOURCE.replace('3.0f', '4.0f'), KERNEL);
    expect(countEntries()).toBe(before + 1);
  });

  it('a corrupt cache entry falls back to compiling instead of throwing', () => {
    const uniq = SOURCE.replace(KERNEL, KERNEL + '_corrupt');
    const name = KERNEL + '_corrupt';
    program.compileToPTX(uniq, name);
    program.clearProgramCache();

    for (const bucket of readdirSync(cacheDir)) {
      for (const f of readdirSync(join(cacheDir, bucket))) {
        if (f.endsWith('.ptx')) rmSync(join(cacheDir, bucket, f));
      }
    }
    const ptx = program.compileToPTX(uniq, name);
    expect(ptx.length).toBeGreaterThan(0);
  });
});
