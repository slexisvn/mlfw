import { CPUTarget, WasmTarget, CUDATarget, WebGPUTarget } from 'mlfw/index.js';
import type { TargetName } from '../protocol.js';

export type TargetFactory = () => unknown;

export const TARGET_FACTORIES: Record<TargetName, TargetFactory> = {
  cpu: CPUTarget,
  wasm: WasmTarget,
  cuda: CUDATarget,
  webgpu: WebGPUTarget,
};
