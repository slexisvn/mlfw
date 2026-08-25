import type { TargetName } from '../protocol.js';

export type TargetNote = { id: TargetName; label: string; note: string; runsHere: boolean };

export const TARGETS: TargetNote[] = [
  { id: 'cpu', label: 'CPU (JS)', note: 'plain JavaScript, runs here', runsHere: true },
  { id: 'wasm', label: 'WebAssembly', note: 'compiled to wasm, runs here', runsHere: true },
  { id: 'cuda', label: 'CUDA', note: 'NVIDIA C++, needs a driver to run', runsHere: false },
  { id: 'webgpu', label: 'WebGPU', note: 'WGSL shaders, runs on your GPU', runsHere: true },
];

const BY_ID = new Map(TARGETS.map(target => [target.id, target]));

export function targetNote(id: TargetName): TargetNote {
  return BY_ID.get(id) as TargetNote;
}
