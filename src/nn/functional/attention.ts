import * as ops from '../../tensor/ops/ops.js';
import { transpose as viewTranspose } from '../../tensor/ops/ops.js';
import { SymbolicTensor } from '../../tracing/symbolic_tensor.js';
import { getActiveTracer } from '../../tracing/tracer.js';
import { empty, full } from '../../tensor/factory/creation_ops.js';
import { dropout } from './dropout.js';
import type { NNTensor, OptionalTensor } from '../types.js';


function _softmax(x: NNTensor, dim: number): NNTensor {
  if (x instanceof SymbolicTensor || x.isSymbolic) {
    return ops.softmax(x, dim) as NNTensor;
  }
  const d = dim < 0 ? x.ndim + dim : dim;
  const maxVal = (ops.max(x, d, true) as NNTensor).detach();
  const shifted = ops.sub(x, maxVal) as NNTensor;
  const e = ops.exp(shifted) as NNTensor;
  const s = ops.sum(e, d, true) as NNTensor;
  return ops.div(e, s) as NNTensor;
}

function _transposeLastTwo(t: NNTensor): NNTensor {
  const nd = t.ndim;
  if (t instanceof SymbolicTensor || t.isSymbolic) {
    return ops.transpose(t, nd - 2, nd - 1) as NNTensor;
  }
  return viewTranspose(t, nd - 2, nd - 1) as NNTensor;
}

function _generateCausalMask(L: number, S: number): NNTensor {
  const mask = empty([L, S]);
  const data = mask._impl.storage.data!;
  const offset = S - L;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < S; j++) {
      data[i * S + j] = j <= i + offset ? 0 : -Infinity;
    }
  }
  return mask as NNTensor;
}

export function scaled_dot_product_attention(query: NNTensor, key: NNTensor, value: NNTensor, attnMask: OptionalTensor = null, dropoutP = 0, isCausal = false, training = false): NNTensor {
  const E = query.shape[query.ndim - 1];
  const L = query.shape[query.ndim - 2];
  const S = key.shape[key.ndim - 2];

  const tracer = getActiveTracer();
  const fusable = tracer && !attnMask && !(dropoutP > 0 && training)
    && query instanceof SymbolicTensor && key instanceof SymbolicTensor && value instanceof SymbolicTensor
    && query.ndim === 4;
  if (fusable) {
    return tracer.recordOp('scaled_dot_product_attention', [query, key, value], { scale: 1 / Math.sqrt(E), causal: isCausal }) as unknown as NNTensor;
  }

  const scale = full([], 1 / Math.sqrt(E));
  const kT = _transposeLastTwo(key);
  let scores = ops.matmul(query, kT) as NNTensor;
  scores = ops.mul(scores, scale) as NNTensor;

  if (isCausal) {
    scores = ops.add(scores, _generateCausalMask(L, S)) as NNTensor;
  }
  if (attnMask) {
    scores = ops.add(scores, attnMask) as NNTensor;
  }

  let weights = _softmax(scores, -1);

  if (dropoutP > 0 && training) {
    weights = dropout(weights, dropoutP, true) as NNTensor;
  }

  return ops.matmul(weights, value) as NNTensor;
}
