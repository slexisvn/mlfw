import type { Layout } from './types.js';
import type { Operation } from './operation.js';

export type LayoutTarget = {
  isGPU(): boolean;
  isCPU(): boolean;
  preferredConvLayout?: Readonly<{ order: readonly number[]; block?: Readonly<{ dim: number; factor: number }> | null }> | null;
  cacheLineBytes?: number;
};

export class LayoutPreference {
  inputs: readonly (Layout | null)[];
  outputs: readonly (Layout | null)[];
  cost: number;

  constructor(inputs: readonly (Layout | null)[], outputs: readonly (Layout | null)[], cost = 0) {
    this.inputs = inputs;
    this.outputs = outputs;
    this.cost = cost;
  }
}

export type InferLayoutFn = (op: Operation, target: LayoutTarget) => LayoutPreference | null;
