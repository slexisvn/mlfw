import * as ops from '../../tensor/ops/ops.js';
import { transpose as viewTranspose } from '../../tensor/ops/ops.js';
import { SymbolicTensor } from '../../tracing/symbolic_tensor.js';
import { getActiveTracer } from '../../tracing/tracer.js';
import { empty, full } from '../../tensor/factory/creation_ops.js';
import { dropout } from './dropout.js';


function _softmax(x, dim) {
  if (x instanceof SymbolicTensor || x.isSymbolic) {
    return ops.softmax(x, dim);
  }
  const d = dim < 0 ? x.ndim + dim : dim;
  const maxVal = ops.max(x, d, true).detach();
  const shifted = ops.sub(x, maxVal);
  const e = ops.exp(shifted);
  const s = ops.sum(e, d, true);
  return ops.div(e, s);
}

function _transposeLastTwo(t) {
  const nd = t.ndim;
  if (t instanceof SymbolicTensor || t.isSymbolic) {
    return ops.transpose(t, nd - 2, nd - 1);
  }
  return viewTranspose(t, nd - 2, nd - 1);
}

function _generateCausalMask(L, S) {
  const mask = empty([L, S]);
  const data = mask._impl.storage.data;
  const offset = S - L;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < S; j++) {
      data[i * S + j] = j <= i + offset ? 0 : -Infinity;
    }
  }
  return mask;
}

export function scaled_dot_product_attention(query, key, value, attnMask = null, dropoutP = 0, isCausal = false, training = false) {
  const E = query.shape[query.ndim - 1];
  const L = query.shape[query.ndim - 2];
  const S = key.shape[key.ndim - 2];

  const tracer = getActiveTracer();
  const fusable = tracer && !attnMask && !(dropoutP > 0 && training)
    && query instanceof SymbolicTensor && key instanceof SymbolicTensor && value instanceof SymbolicTensor
    && query.ndim === 4;
  if (fusable) {
    return tracer.recordOp('scaled_dot_product_attention', [query, key, value], { scale: 1 / Math.sqrt(E), causal: isCausal });
  }

  const scale = full([], 1 / Math.sqrt(E));
  const kT = _transposeLastTwo(key);
  let scores = ops.matmul(query, kT);
  scores = ops.mul(scores, scale);

  if (isCausal) {
    scores = ops.add(scores, _generateCausalMask(L, S));
  }
  if (attnMask) {
    scores = ops.add(scores, attnMask);
  }

  let weights = _softmax(scores, -1);

  if (dropoutP > 0 && training) {
    weights = dropout(weights, dropoutP, true);
  }

  return ops.matmul(weights, value);
}
