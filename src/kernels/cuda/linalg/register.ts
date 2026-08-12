import { Library } from '../../../dispatcher/library.js';
import { DispatchKey } from '../../../dispatcher/dispatch_key.js';
import { ensureLinalgSchemas } from '../../defs/linalg_defs.js';
import {
  gpuSvd, gpuEigh, gpuCholesky, gpuSolve, gpuLstsq, gpuInv, gpuPinv, gpuDet, gpuCov,
} from './cusolver_linalg_ops.js';

const _CUDA_KERNELS = {
  svd: gpuSvd,
  eigh: gpuEigh,
  cholesky: gpuCholesky,
  inv: gpuInv,
  pinv: gpuPinv,
  det: gpuDet,
  cov: gpuCov,
  solve: gpuSolve,
  lstsq: gpuLstsq,
};

let _registered = false;

export function registerCudaLinalg(): void {
  if (_registered) return;
  _registered = true;
  ensureLinalgSchemas();
  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(_CUDA_KERNELS)) {
    implLib.impl(name, DispatchKey.GPU, fn);
  }
}
