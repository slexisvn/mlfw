import { Library } from '../../../dispatcher/library.js';
import { DispatchKey } from '../../../dispatcher/dispatch_key.js';
import { ensureMlSchemas } from '../../defs/ml_defs.js';
import { wasmKmeans, wasmKmeansPredict } from './kmeans.js';
import { wasmKnnPredict } from './knn.js';
import { wasmElasticNet } from './elastic_net.js';

const _SIMD_KERNELS = {
  kmeans: wasmKmeans,
  kmeans_predict: wasmKmeansPredict,
  knn_predict: wasmKnnPredict,
  elastic_net: wasmElasticNet,
};

let _registered = false;

export function registerWasmMl() {
  const summary = { enabled: Object.keys(_SIMD_KERNELS) };
  if (_registered) return summary;
  _registered = true;
  ensureMlSchemas();
  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(_SIMD_KERNELS)) {
    implLib.impl(name, DispatchKey.WASM, fn);
  }
  return summary;
}
