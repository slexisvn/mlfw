import * as ops from '../../tensor/ops/ops.js';

export function embedding(weight, indices) {
  return ops.embedding(weight, indices);
}
