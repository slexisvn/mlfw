import { Library } from '../../dispatcher/library.js';

export const NUMERIC_SCHEMAS = [
  'qr(Tensor input) -> (Tensor, Tensor)',
  'fft(Tensor input) -> Tensor',
  'ifft(Tensor input) -> Tensor',
];

let _defined = false;

export function ensureNumericSchemas() {
  if (_defined) return;
  _defined = true;
  const defLib = new Library('mlc', 'DEF');
  for (const schema of NUMERIC_SCHEMAS) defLib.def(schema);
}
