import { Library } from '../../../dispatcher/library.js';
import { DispatchKey } from '../../../dispatcher/dispatch_key.js';
import { ensureNumericSchemas } from '../../defs/numeric_defs.js';
import { cpuQr, cpuFft, cpuIfft } from './ops.js';

const _CPU_KERNELS = {
  qr: cpuQr,
  fft: cpuFft,
  ifft: cpuIfft,
};

const _HOST_KEYS = [DispatchKey.CPU, DispatchKey.WASM];

let _registered = false;

export function registerCpuNumeric(): void {
  if (_registered) return;
  _registered = true;
  ensureNumericSchemas();
  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(_CPU_KERNELS)) {
    for (const key of _HOST_KEYS) implLib.impl(name, key, fn);
  }
}
