import { computePartitionIO } from './partition_core.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';

export abstract class OpGroup {
  id: number;
  ops: Operation[];
  opSet: Set<Operation>;
  private _inputValues: Value[] | null;
  private _outputValues: Value[] | null;

  constructor(id: number) {
    this.id = id;
    this.ops = [];
    this.opSet = new Set();
    this._inputValues = null;
    this._outputValues = null;
  }

  addOp(op: Operation): boolean {
    if (this.opSet.has(op)) return false;
    this.ops.push(op);
    this.opSet.add(op);
    this.invalidateIO();
    return true;
  }

  invalidateIO(): void {
    this._inputValues = null;
    this._outputValues = null;
  }

  hasOp(op: Operation): boolean {
    return this.opSet.has(op);
  }

  computeIO(): void {
    if (this._inputValues && this._outputValues) return;
    const { inputs, outputs } = computePartitionIO(this.opSet, this.ops);
    this._inputValues = inputs;
    this._outputValues = outputs;
  }

  getInputValues(): Value[] {
    this.computeIO();
    return this._inputValues as Value[];
  }

  getOutputValues(): Value[] {
    this.computeIO();
    return this._outputValues as Value[];
  }

  get size(): number {
    return this.ops.length;
  }
}
