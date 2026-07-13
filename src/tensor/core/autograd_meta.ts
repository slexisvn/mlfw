import type { Tensor } from './tensor.js';

export class AutogradMeta {
  private _grad: Tensor | null;
  private _gradFn: unknown | null;
  private _outputNr: number;
  private _requiresGrad: boolean;
  private _retainGrad: boolean;
  private _gradAccumulator: WeakRef<object> | object | null;
  private _versionAtCreation: number;

  constructor() {
    this._grad = null;
    this._gradFn = null;
    this._outputNr = 0;
    this._requiresGrad = false;
    this._retainGrad = false;
    this._gradAccumulator = null;
    this._versionAtCreation = 0;
  }

  get grad(): Tensor | null {
    return this._grad;
  }

  set grad(tensor: Tensor | null) {
    this._grad = tensor;
  }

  get gradFn(): unknown | null {
    return this._gradFn;
  }

  setGradFn(node: unknown, outputNr?: number) {
    this._gradFn = node;
    this._outputNr = outputNr ?? 0;
  }

  get outputNr(): number {
    return this._outputNr;
  }

  get requiresGrad(): boolean {
    return this._requiresGrad;
  }

  set requiresGrad(flag: boolean) {
    this._requiresGrad = flag;
  }

  get retainGrad(): boolean {
    return this._retainGrad;
  }

  set retainGrad(flag: boolean) {
    this._retainGrad = flag;
  }

  get isLeaf(): boolean {
    return this._gradFn === null;
  }

  get versionAtCreation(): number {
    return this._versionAtCreation;
  }

  set versionAtCreation(v: number) {
    this._versionAtCreation = v;
  }

  getGradAccumulator(): object | null {
    if (this._gradAccumulator) {
      const acc = this._gradAccumulator instanceof WeakRef ? this._gradAccumulator.deref() : this._gradAccumulator;
      if (acc) return acc;
    }
    return null;
  }

  setGradAccumulator(acc: object) {
    this._gradAccumulator = typeof WeakRef !== 'undefined' ? new WeakRef(acc) : acc;
  }

  clearGrad() {
    this._grad = null;
  }

  clearGradFn() {
    this._gradFn = null;
    this._outputNr = 0;
  }
}
