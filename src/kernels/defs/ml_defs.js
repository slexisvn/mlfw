import { Library } from '../../dispatcher/library.js';

export const ML_SCHEMAS = [
  'kmeans(Tensor x, int n_clusters, int max_iter, int n_init, int seed) -> (Tensor, Tensor, Tensor)',
  'kmeans_predict(Tensor x, Tensor centers) -> Tensor',
  'knn_predict(Tensor x_train, Tensor y_train, Tensor x_query, int n_neighbors, bool classify) -> Tensor',
  'gaussian_nb_fit(Tensor x, Tensor y) -> (Tensor, Tensor, Tensor, Tensor)',
  'gaussian_nb_predict(Tensor x, Tensor means, Tensor variances, Tensor priors, Tensor classes) -> Tensor',
  'elastic_net(Tensor x, Tensor y, float alpha, float l1_ratio, int max_iter, float tol, bool fit_intercept) -> (Tensor, Tensor)',
  'decision_tree_fit(Tensor x, Tensor y, int max_depth, int min_split, int min_leaf, int max_features, bool classify, int seed) -> (Tensor, Tensor, Tensor, Tensor, Tensor)',
  'decision_tree_predict(Tensor x, Tensor feature, Tensor threshold, Tensor left, Tensor right, Tensor value) -> Tensor',
];

let _defined = false;

export function ensureMlSchemas() {
  if (_defined) return;
  _defined = true;
  const defLib = new Library('mlc', 'DEF');
  for (const schema of ML_SCHEMAS) defLib.def(schema);
}
