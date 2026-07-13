import type { DType } from '../tensor/types/dtype.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { GradInputList, GradOutputList, InputMetadata, OpArgs } from './types.js';

let _nodeIdCounter = 0;

export class Edge {
  readonly node: AutogradNode | null;
  readonly inputNr: number;

  constructor(node: AutogradNode | null, inputNr: number) {
    this.node = node;
    this.inputNr = inputNr;
  }
}

export class AutogradNode {
  private readonly _id: number;
  private readonly _numInputs: number;
  private readonly _nextEdges: Array<Edge | null>;
  private _savedTensors: Tensor[];
  private _inputMetadata: Array<InputMetadata | undefined>;
  private _opArgs: OpArgs;

  constructor(numInputs?: number) {
    this._id = _nodeIdCounter++;
    this._numInputs = numInputs || 0;
    this._nextEdges = [];
    this._savedTensors = [];
    this._inputMetadata = [];
    this._opArgs = null;
  }

  setOpArgs(args: OpArgs) {
    this._opArgs = args;
  }

  opArgs(): OpArgs {
    return this._opArgs;
  }

  get id(): number {
    return this._id;
  }

  get numInputs(): number {
    return this._numInputs;
  }

  get nextEdges(): readonly (Edge | null)[] {
    return this._nextEdges;
  }

  addNextEdge(node: AutogradNode | null, inputNr: number) {
    this._nextEdges.push(new Edge(node, inputNr));
  }

  setNextEdge(index: number, node: AutogradNode | null, inputNr: number) {
    while (this._nextEdges.length <= index) this._nextEdges.push(null);
    this._nextEdges[index] = new Edge(node, inputNr);
  }

  saveTensor(tensor: Tensor) {
    this._savedTensors.push(tensor);
  }

  savedTensors(): readonly Tensor[] {
    return this._savedTensors;
  }

  saveInputMetadata(index: number, shape: readonly number[], dtype: DType) {
    this._inputMetadata[index] = { shape, dtype };
  }

  inputMetadata(index: number): InputMetadata | null {
    return this._inputMetadata[index] || null;
  }

  apply(_gradOutputs: GradOutputList): GradInputList | null {
    throw new Error(`${this.name()}.apply() not implemented`);
  }

  name(): string {
    return this.constructor.name;
  }

  releaseVariables(): void {
    this._savedTensors = [];
    this._inputMetadata = [];
  }
}
