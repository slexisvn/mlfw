const RUNS_HERE = null;

export const TARGETS = [
  {
    id: 'cpu', label: 'CPU (JS)', note: 'plain JavaScript, runs here',
    kernelLanguage: 'javascript', skipReason: RUNS_HERE,
  },
  {
    id: 'wasm', label: 'WebAssembly', note: 'compiled to wasm, runs here',
    kernelLanguage: 'wat', skipReason: RUNS_HERE,
  },
  {
    id: 'cuda', label: 'CUDA', note: 'NVIDIA C++, needs a driver to run',
    kernelLanguage: 'cpp',
    skipReason: 'CUDA kernels need a native driver — the source above is real, but nothing here can launch it.',
  },
  {
    id: 'webgpu', label: 'WebGPU', note: 'WGSL shaders, runs on your GPU',
    kernelLanguage: 'wgsl', skipReason: RUNS_HERE,
  },
] as const;

export type TargetNote = (typeof TARGETS)[number];

export type TargetName = TargetNote['id'];

const BY_ID = new Map<string, TargetNote>(TARGETS.map(target => [target.id, target]));

export function targetNote(id: TargetName): TargetNote {
  return BY_ID.get(id) as TargetNote;
}
