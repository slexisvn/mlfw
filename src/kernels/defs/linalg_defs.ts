import { Library } from '../../dispatcher/library.js';

export const LINALG_SCHEMAS = [
  'svd(Tensor input) -> (Tensor, Tensor, Tensor)',
  'eigh(Tensor input) -> (Tensor, Tensor)',
  'cholesky(Tensor input) -> Tensor',
  'inv(Tensor input) -> Tensor',
  'pinv(Tensor input) -> Tensor',
  'det(Tensor input) -> Tensor',
  'cov(Tensor input) -> Tensor',
  'solve(Tensor a, Tensor b) -> Tensor',
  'lstsq(Tensor a, Tensor b) -> Tensor',
];

let _defined = false;

export function ensureLinalgSchemas(): void {
  if (_defined) return;
  _defined = true;
  const defLib = new Library('mlc', 'DEF');
  for (const schema of LINALG_SCHEMAS) defLib.def(schema);
}
