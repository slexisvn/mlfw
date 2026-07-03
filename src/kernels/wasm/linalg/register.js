import { Library } from '../../../dispatcher/library.js';
import { DispatchKey } from '../../../dispatcher/dispatch_key.js';
import { ensureLinalgSchemas } from '../../defs/linalg_defs.js';
import { wasmSvd } from './svd.js';

const _SIMD_KERNELS = {
  svd: wasmSvd,
};

const _SKIPPED = {
  eigh: 'Jacobi rotation sweeps use strided column access; not SIMD-friendly',
  cholesky: 'sequential dependent updates; not vectorizable at f64x2 width',
  solve: 'triangular back-substitution is inherently sequential',
  cov: 'single O(n*d) pass dominated by the eigen-free reduction',
};

let _registered = false;

export function registerWasmLinalg() {
  const summary = { enabled: Object.keys(_SIMD_KERNELS), skipped: Object.keys(_SKIPPED) };
  if (_registered) return summary;
  _registered = true;
  ensureLinalgSchemas();
  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(_SIMD_KERNELS)) {
    implLib.impl(name, DispatchKey.WASM, fn);
  }
  return summary;
}
