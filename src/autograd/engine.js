
import { ones as _ones } from '../tensor/factory/creation_ops.js';
import { add as _add, sum as _sum } from '../tensor/ops/ops.js';

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

    const rawGradInputs = node.apply(grads);

    // Reduce gradients to match input shapes (handle broadcasting)
    let gradInputs = rawGradInputs;
    if (rawGradInputs) {
      gradInputs = rawGradInputs.map((g, i) => {
        if (!g) return g;
        const meta = node.inputMetadata(i);
        if (!meta) return g;
        return _reduceBroadcastGrad(g, meta.shape);
      });
    }

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

/**
 * Reduce gradient to match the input shape when broadcasting was used.
 * e.g., grad shape [2,3] → input shape [2,1]: sum along axis 1, keepdim.
 * e.g., grad shape [3,4] → input shape [4]: sum along axis 0.
 */
function _reduceBroadcastGrad(grad, inputShape) {
  const gradShape = grad.shape;
  if (gradShape.length === inputShape.length &&
      gradShape.every((s, i) => s === inputShape[i])) {
    return grad; // shapes match, no reduction needed
  }

  let result = grad;

  // If input has fewer dimensions, sum leading dims
  const dimDiff = gradShape.length - inputShape.length;
  for (let i = 0; i < dimDiff; i++) {
    result = _sum(result, 0, false);
  }

  // Now same rank — sum along dims where input was 1 (broadcast dims)
  // Must go from last to first to keep indices stable
  for (let i = inputShape.length - 1; i >= 0; i--) {
    if (inputShape[i] === 1 && result.shape[i] !== 1) {
      result = _sum(result, i, true);
    }
  }

  return result;
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
