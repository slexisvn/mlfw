import { Library } from '../../../dispatcher/library.js';
import { DispatchKey } from '../../../dispatcher/dispatch_key.js';
import { ensureLinalgSchemas } from '../../linalg_defs.js';
import {
  cpuCholesky, cpuSolve, cpuLstsq, cpuInv, cpuPinv, cpuDet, cpuCov, cpuEigh, cpuSvd,
} from './ops.js';

const _CPU_KERNELS = {
  svd: cpuSvd,
  eigh: cpuEigh,
  cholesky: cpuCholesky,
  inv: cpuInv,
  pinv: cpuPinv,
  det: cpuDet,
  cov: cpuCov,
  solve: cpuSolve,
  lstsq: cpuLstsq,
};

const _HOST_KEYS = [DispatchKey.CPU, DispatchKey.WASM];

let _registered = false;

export function registerCpuLinalg() {
  if (_registered) return;
  _registered = true;
  ensureLinalgSchemas();
  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(_CPU_KERNELS)) {
    for (const key of _HOST_KEYS) implLib.impl(name, key, fn);
  }
}
