import { Library } from '../../../dispatcher/library.js';
import { DispatchKey } from '../../../dispatcher/dispatch_key.js';
import { ensureMlSchemas } from '../../ml_defs.js';
import { cpuKmeans, cpuKmeansPredict } from './kmeans.js';
import { cpuKnnPredict } from './knn.js';
import { cpuGaussianNbFit, cpuGaussianNbPredict } from './gaussian_nb.js';
import { cpuElasticNet } from './elastic_net.js';
import { cpuDecisionTreeFit, cpuDecisionTreePredict } from './tree.js';

const _CPU_KERNELS = {
  kmeans: cpuKmeans,
  kmeans_predict: cpuKmeansPredict,
  knn_predict: cpuKnnPredict,
  gaussian_nb_fit: cpuGaussianNbFit,
  gaussian_nb_predict: cpuGaussianNbPredict,
  elastic_net: cpuElasticNet,
  decision_tree_fit: cpuDecisionTreeFit,
  decision_tree_predict: cpuDecisionTreePredict,
};

const _HOST_KEYS = [DispatchKey.CPU, DispatchKey.WASM];
const _GPU_KEYS = [DispatchKey.GPU, DispatchKey.CUSTOM_0];

function unsupported(name) {
  return () => {
    throw new Error(`ml.${name}: scalar-iterative algorithm runs on CPU/WASM only; GPU/WebGPU not supported (no performance benefit)`);
  };
}

let _registered = false;

export function registerCpuMl() {
  if (_registered) return;
  _registered = true;
  ensureMlSchemas();
  const implLib = new Library('mlc', 'IMPL');
  for (const [name, fn] of Object.entries(_CPU_KERNELS)) {
    for (const key of _HOST_KEYS) implLib.impl(name, key, fn);
    for (const key of _GPU_KEYS) implLib.impl(name, key, unsupported(name));
  }
}
