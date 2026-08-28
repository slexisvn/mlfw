import { defineHostOps } from './host_ops.js';

export const NUMERIC_SCHEMAS = [
  'qr(Tensor input) -> (Tensor, Tensor)',
  'fft(Tensor input) -> Tensor',
  'ifft(Tensor input) -> Tensor',
];

export const ensureNumericSchemas = defineHostOps({
  devices: ['cpu', 'wasm'],
  schemas: NUMERIC_SCHEMAS,
});
