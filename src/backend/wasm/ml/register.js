import { Library } from '../../../dispatcher/library.js';
import { DispatchKey } from '../../../dispatcher/dispatch_key.js';
import { ensureMlSchemas } from '../../ml_defs.js';
import { wasmKmeans, wasmKmeansPredict } from './kmeans.js';
import { wasmKnnPredict } from './knn.js';
import { wasmElasticNet } from './elastic_net.js';

const _SIMD_KERNELS = {
  kmeans: wasmKmeans,
  kmeans_predict: wasmKmeansPredict,
  knn_predict: wasmKnnPredict,
  elastic_net: wasmElasticNet,
};

const _SKIPPED = {
  gaussian_nb_fit: 'single O(n*d) pass; no measurable SIMD benefit',
  gaussian_nb_predict: 'single O(n*d) pass; no measurable SIMD benefit',
  decision_tree_fit: 'branch-heavy split search; SIMD not beneficial',
  decision_tree_predict: 'pointer-chasing traversal; not vectorizable',
};

let _registered = false;

export function registerWasmMl() {
  const summary = { enabled: Object.keys(_SIMD_KERNELS), skipped: Object.keys(_SKIPPED) };
  if (_registered) return summary;
  _registered = true;
  ensureMlSchemas();
  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(_SIMD_KERNELS)) {
    implLib.impl(name, DispatchKey.WASM, fn);
  }
  if (process.env.MLFW_WASM_SIMD) {
    for (const name of summary.enabled) {
      console.info(`[mlfw] ml.${name}: WASM+SIMD kernel registered on WASM device`);
    }
    for (const [name, why] of Object.entries(_SKIPPED)) {
      console.info(`[mlfw] ml.${name}: WASM+SIMD skipped (${why}); JS kernel kept on WASM device`);
    }
  }
  return summary;
}
