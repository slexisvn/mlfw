import { GradAccumulator } from './accumulator.js';
import { ones as _ones } from '../tensor/factory/creation_ops.js';
import { add as _add } from '../tensor/ops/ops.js';

export function backward(rootTensor, gradOutput) {
  const rootGradFn = rootTensor.gradFn;
  if (!rootGradFn) {
    throw new Error('Cannot call backward on a tensor that does not require grad');
  }

  let rootGrad = gradOutput;
  if (!rootGrad) {
    if (rootTensor.numel !== 1) {
      throw new Error('grad must be specified for non-scalar tensors');
    }
    rootGrad = _ones(rootTensor.shape, { dtype: rootTensor.dtype, device: rootTensor.device });
  }

  const depCount = new Map();
  const visited = new Set();

  _countDeps(rootGradFn, depCount, visited);

  const gradMap = new Map();
  gradMap.set(rootGradFn.id, [rootGrad]);

  const ready = [];
  for (const [node, count] of depCount) {
    if (count === 0) ready.push(node);
  }

  const order = [];
  while (ready.length > 0) {
    const node = ready.pop();
    order.push(node);

    for (const edge of node.nextEdges) {
      if (!edge || !edge.node) continue;
      const child = edge.node;
      const newCount = depCount.get(child) - 1;
      depCount.set(child, newCount);
      if (newCount === 0) ready.push(child);
    }
  }

  for (const node of order) {
    const grads = gradMap.get(node.id);
    if (!grads) continue;

    const gradInputs = node.apply(grads);
    node.releaseVariables();

    if (!gradInputs) continue;

    const edges = node.nextEdges;
    for (let i = 0; i < edges.length; i++) {
      const edge = edges[i];
      if (!edge || !edge.node) continue;
      if (i >= gradInputs.length || !gradInputs[i]) continue;

      const childId = edge.node.id;
      const existing = gradMap.get(childId);
      if (existing) {
        existing[edge.inputNr] = existing[edge.inputNr]
          ? _add(existing[edge.inputNr], gradInputs[i])
          : gradInputs[i];
      } else {
        const arr = [];
        arr[edge.inputNr] = gradInputs[i];
        gradMap.set(childId, arr);
      }
    }
  }
}

function _countDeps(root, depCount, visited) {
  const queue = [root];
  depCount.set(root, 0);

  while (queue.length > 0) {
    const node = queue.shift();
    if (visited.has(node.id)) continue;
    visited.add(node.id);

    for (const edge of node.nextEdges) {
      if (!edge || !edge.node) continue;
      const child = edge.node;
      if (!depCount.has(child)) {
        depCount.set(child, 0);
      }
      depCount.set(child, depCount.get(child) + 1);

      if (!visited.has(child.id)) {
        queue.push(child);
      }
    }
  }
}
