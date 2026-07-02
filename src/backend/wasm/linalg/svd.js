import { hostMatrix, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { svdHost } from '../../cpu/linalg/svd.js';
import { cpuSvd } from '../../cpu/linalg/ops.js';
import { simdModule } from '../simd/runtime.js';
import { gramSymWat } from '../simd/kernels.js';

const GRAM_MIN_WORK = 500000;

function gramSym(mat, m, len) {
  const mod = simdModule('gram_sym', gramSymWat, 'gram_sym');
  mod.reset();
  const mp = mod.allocF64(m * len);
  const gp = mod.allocF64(m * m);
  mod.writeF64(mp, mat);
  mod.run(mp, m, len, gp);
  return Float64Array.from(mod.f64(gp, m * m));
}

function wasmGram(a, rows, cols, transposeLeft) {
  if (transposeLeft) {
    const acol = new Float64Array(cols * rows);
    for (let t = 0; t < rows; t++) {
      for (let i = 0; i < cols; i++) acol[i * rows + t] = a[t * cols + i];
    }
    return gramSym(acol, cols, rows);
  }
  return gramSym(a, rows, cols);
}

export function wasmSvd(_keySet, a) {
  const { data, rows, cols } = hostMatrix(a);
  const gramWork = Math.max(rows, cols) * Math.min(rows, cols) * Math.min(rows, cols);
  if (gramWork < GRAM_MIN_WORK) return cpuSvd(_keySet, a);
  const { U, S, V, k } = svdHost(data, rows, cols, undefined, wasmGram);
  return [
    toHostTensor(U, [rows, k], a.dtype, a.device),
    toHostTensor(S, [k], a.dtype, a.device),
    toHostTensor(V, [cols, k], a.dtype, a.device),
  ];
}
