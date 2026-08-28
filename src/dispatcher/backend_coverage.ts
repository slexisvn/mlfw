import { DispatchKey, deviceForBackendKey } from './dispatch_key.js';
import type { DispatchKeyValue } from './dispatch_key.js';
import { KernelFunction } from './boxing.js';
import { dispatcher } from './dispatcher.js';
import { jitKernelFor } from './jit_dispatch.js';
import { canBuildMappedOp } from '../tensor/ops/ir_mapping.js';
import type { OperatorHandle } from './operator_handle.js';

export const BACKEND_COVERAGE_KEYS: readonly DispatchKeyValue[] = Object.freeze([
  DispatchKey.CPU,
  DispatchKey.GPU,
  DispatchKey.WASM,
  DispatchKey.CUSTOM_0,
]);

export function isBackendCoverageKey(key: DispatchKeyValue): boolean {
  return BACKEND_COVERAGE_KEYS.indexOf(key) >= 0;
}

export function coverageGap(handle: OperatorHandle, key: DispatchKeyValue): string | null {
  if (handle.entry.hasKernel(key)) return null;
  const devices = handle.entry.devices;
  const device = deviceForBackendKey(key);
  if (devices) {
    if (devices.has(device)) {
      return `Op '${handle.name}' runs on ${device}, but its ${device} kernels have not been registered; load the ${device} runtime before calling it`;
    }
    return `Op '${handle.name}' has no ${device} implementation; it runs on ${[...devices].join(', ')} only`;
  }
  return `Op '${handle.name}' has no ${device} kernel, and the compiler has no graph-IR lowering for '${handle.name}' to generate one; register a ${device} kernel, or declare the devices it runs on`;
}

function coverOperator(handle: OperatorHandle): void {
  if (handle.entry.devices) return;
  for (const key of BACKEND_COVERAGE_KEYS) {
    if (handle.entry.hasKernel(key)) continue;
    const kernel = jitKernelFor(handle.name, key);
    if (kernel) handle.entry.registerKernel(key, KernelFunction.fromUnboxed(kernel));
  }
}

function explainMissingKernel(handle: OperatorHandle, key: DispatchKeyValue): string | null {
  if (!isBackendCoverageKey(key)) return null;
  if (handle.entry.devices === null && canBuildMappedOp(handle.name)) return null;
  return coverageGap(handle, key);
}

export function installBackendCoverage(): void {
  dispatcher.setMissingKernelExplainer(explainMissingKernel);
  dispatcher.onOpDefined(coverOperator);
}
