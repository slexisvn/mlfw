
import { ones as _ones } from '../tensor/factory/creation_ops.js';
import { add as _add, sum as _sum } from '../tensor/ops/ops.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { AutogradNode } from './node.js';
import type { GradInputList, GradOutputList } from './types.js';

export function backward(rootTensor: Tensor, gradOutput?: Tensor): void {
  const rootGradFn = rootTensor.gradFn as AutogradNode | null;
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

  const depCount = new Map<AutogradNode, number>();
  const visited = new Set<number>();

  _countDeps(rootGradFn, depCount, visited);

  const gradMap = new Map<number, GradInputList>();
  gradMap.set(rootGradFn.id, [rootGrad]);

  const ready: AutogradNode[] = [];
  for (const [node, count] of depCount) {
    if (count === 0) ready.push(node);
  }

  const order: AutogradNode[] = [];
  while (ready.length > 0) {
    const node = ready.pop();
    if (!node) continue;
    order.push(node);

    for (const edge of node.nextEdges) {
      if (!edge || !edge.node) continue;
      const child = edge.node;
      const newCount = depCount.get(child)! - 1;
      depCount.set(child, newCount);
      if (newCount === 0) ready.push(child);
    }
  }

  for (const node of order) {
    const grads = gradMap.get(node.id) as GradOutputList | undefined;
    if (!grads) continue;

    const rawGradInputs = node.apply(grads);

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
      const gradInput = gradInputs[i]!;

      const childId = edge.node.id;
      const existing = gradMap.get(childId);
      if (existing) {
        const existingGrad = existing[edge.inputNr];
        existing[edge.inputNr] = existing[edge.inputNr]
          ? _add(existingGrad!, gradInput)
          : gradInput;
      } else {
        const arr: GradInputList = [];
        arr[edge.inputNr] = gradInput;
        gradMap.set(childId, arr);
      }
    }
  }
}

function _reduceBroadcastGrad(grad: Tensor, inputShape: readonly number[]): Tensor {
  const gradShape = grad.shape;
  if (gradShape.length === inputShape.length &&
      gradShape.every((s, i) => s === inputShape[i])) {
    return grad;
  }

  let result = grad;

  const dimDiff = gradShape.length - inputShape.length;
  for (let i = 0; i < dimDiff; i++) {
    result = _sum(result, 0, false);
  }

  for (let i = inputShape.length - 1; i >= 0; i--) {
    if (inputShape[i] === 1 && result.shape[i] !== 1) {
      result = _sum(result, i, true);
    }
  }

  return result;
}

function _countDeps(root: AutogradNode, depCount: Map<AutogradNode, number>, visited: Set<number>): void {
  const queue = [root];
  depCount.set(root, 0);

  while (queue.length > 0) {
    const node = queue.shift()!;
    if (visited.has(node.id)) continue;
    visited.add(node.id);

    for (const edge of node.nextEdges) {
      if (!edge || !edge.node) continue;
      const child = edge.node;
      if (!depCount.has(child)) {
        depCount.set(child, 0);
      }
      depCount.set(child, depCount.get(child)! + 1);

      if (!visited.has(child.id)) {
        queue.push(child);
      }
    }
  }
}
