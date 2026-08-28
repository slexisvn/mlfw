import { defineHostOps } from './host_ops.js';

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

export const ensureLinalgSchemas = defineHostOps({
  devices: ['cpu', 'wasm', 'gpu'],
  schemas: LINALG_SCHEMAS,
});
