import { readdirSync, existsSync } from 'fs';
import { join } from 'path';

import { SEARCH_ENV, prependSearchPath } from '../search_path.js';

type LibVariant = { pattern: RegExp; fallback: string; extraDirs?: () => string[] };

export type LibSpec = { win: LibVariant; linux: LibVariant };

const isWin = process.platform === 'win32';

function envRoots(): string[] {
  const roots: string[] = [];
  if (process.env.CUDA_PATH) roots.push(process.env.CUDA_PATH);
  if (!isWin && process.env.CUDA_HOME) roots.push(process.env.CUDA_HOME);
  return roots;
}

function toolkitRoots(): string[] {
  const roots: string[] = [];
  if (isWin) {
    const toolkit = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA';
    if (existsSync(toolkit)) for (const v of readdirSync(toolkit)) roots.push(join(toolkit, v));
  } else {
    const base = '/usr/local';
    if (existsSync(base)) {
      for (const d of readdirSync(base)) {
        if (d === 'cuda' || d.startsWith('cuda-')) roots.push(join(base, d));
      }
    }
  }
  return roots;
}

function libDirs(root: string): string[] {
  if (isWin) return [join(root, 'bin')];
  return [join(root, 'lib64'), join(root, 'targets/x86_64-linux/lib')];
}

function systemDirs(): string[] {
  return isWin ? [] : ['/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu'];
}

function versionKey(name: string): number[] {
  const nums = name.match(/\d+/g);
  if (!nums) return [];
  return nums.map(Number);
}

function compareVersion(a: string, b: string): number {
  const ka = versionKey(a), kb = versionKey(b);
  const n = Math.max(ka.length, kb.length);
  for (let i = 0; i < n; i++) {
    const d = (ka[i] || 0) - (kb[i] || 0);
    if (d !== 0) return d;
  }
  return 0;
}

export function loadCudaLib(spec: LibSpec): string {
  const { pattern, fallback, extraDirs } = isWin ? spec.win : spec.linux;
  const dirs: string[] = [];
  for (const root of [...envRoots(), ...toolkitRoots()]) {
    for (const d of libDirs(root)) dirs.push(d);
  }
  for (const d of systemDirs()) dirs.push(d);
  if (extraDirs) for (const d of extraDirs()) dirs.push(d);
  for (const dir of dirs) {
    if (!existsSync(dir)) continue;
    const matches = readdirSync(dir).filter(f => pattern.test(f) && !f.includes('.alt'));
    if (matches.length > 0) {
      prependSearchPath(dir);
      matches.sort(compareVersion);
      return join(dir, matches.pop()!);
    }
  }
  return fallback;
}

function resolveIncludeDir(): string | null {
  for (const root of [...envRoots(), ...toolkitRoots()]) {
    const inc = join(root, 'include');
    if (existsSync(inc)) return inc;
  }
  return isWin ? null : (existsSync('/usr/local/cuda/include') ? '/usr/local/cuda/include' : null);
}

export const cudaIncludeDir = resolveIncludeDir();

export const DRIVER_SPEC: LibSpec = {
  win: { pattern: /^nvcuda\.dll$/, fallback: 'nvcuda.dll' },
  linux: { pattern: /^libcuda\.so(\.\d+)*$/, fallback: 'libcuda.so.1' },
};
export const NVRTC_SPEC: LibSpec = {
  win: { pattern: /^nvrtc64_\d+_\d+\.dll$/, fallback: 'nvrtc64_120_0.dll' },
  linux: { pattern: /^libnvrtc\.so(\.\d+)*$/, fallback: 'libnvrtc.so.12' },
};
export const CUDART_SPEC: LibSpec = {
  win: { pattern: /^cudart64_\d+\.dll$/, fallback: 'cudart64_12.dll' },
  linux: { pattern: /^libcudart\.so(\.\d+)*$/, fallback: 'libcudart.so.12' },
};
export const CUBLAS_SPEC: LibSpec = {
  win: { pattern: /^cublas64_\d+\.dll$/, fallback: 'cublas64_12.dll' },
  linux: { pattern: /^libcublas\.so(\.\d+)*$/, fallback: 'libcublas.so.12' },
};
export const CUSOLVER_SPEC: LibSpec = {
  win: { pattern: /^cusolver64_\d+\.dll$/, fallback: 'cusolver64_11.dll' },
  linux: { pattern: /^libcusolver\.so(\.\d+)*$/, fallback: 'libcusolver.so.11' },
};
function winCudnnDirs(): string[] {
  const dirs: string[] = [];
  const root = 'C:/Program Files/NVIDIA/CUDNN';
  if (existsSync(root)) {
    for (const ver of readdirSync(root).sort().reverse()) {
      const binBase = join(root, ver, 'bin');
      if (!existsSync(binBase)) continue;
      for (const cd of readdirSync(binBase).filter(d => d.startsWith('12.')).sort().reverse()) {
        dirs.push(join(binBase, cd, 'x64'));
      }
    }
  }
  return dirs;
}

export const CUDNN_SPEC: LibSpec = {
  win: { pattern: /^cudnn64_9\.dll$/, fallback: 'cudnn64_9.dll', extraDirs: winCudnnDirs },
  linux: { pattern: /^libcudnn\.so(\.\d+)*$/, fallback: 'libcudnn.so.9' },
};
