import { AutogradNode } from './node.js';
import { add as _add } from '../tensor/ops/ops.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { GradInputList, GradOutputList } from './types.js';

export class GradAccumulator extends AutogradNode {
  private readonly _variable: WeakRef<Tensor> | { deref: () => Tensor };

  constructor(variable: Tensor) {
    super(0);
    this._variable = typeof WeakRef !== 'undefined' ? new WeakRef(variable) : { deref: () => variable };
  }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const tensor = this._variable.deref();
    if (!tensor) return [];

    const meta = tensor._impl.autogradMeta;
    if (!meta) return [];

    if (meta.grad === null) {
      meta.grad = g;
    } else {
      meta.grad = _add(meta.grad, g);
    }

    return [];
  }

  name(): string {
    return 'GradAccumulator';
  }
}

export function wireInputEdges(node: AutogradNode, inputs: readonly Tensor[]): void {
  for (let i = 0; i < inputs.length; i++) {
    node.saveInputMetadata(i, [...inputs[i].shape], inputs[i].dtype);
    const m = inputs[i]._impl.autogradMeta;
    if (m && m.requiresGrad) {
      if (m.gradFn) node.setNextEdge(i, m.gradFn as AutogradNode, m.outputNr || 0);
      else {
        let acc = m.getGradAccumulator() as unknown as GradAccumulator | null;
        if (!acc) { acc = new GradAccumulator(inputs[i]); m.setGradAccumulator(acc); }
        node.setNextEdge(i, acc, 0);
      }
    } else node.setNextEdge(i, null, 0);
  }
}
