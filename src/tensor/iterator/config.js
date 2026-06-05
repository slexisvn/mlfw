import { TensorIterator } from './iterator.js';

export class TensorIteratorConfig {
  constructor() {
    this._inputs = [];
    this._output = null;
    this._numOutputs = 1;
    this._enforceDtype = null;
    this._promoteInputs = true;
    this._checkShapes = true;
    this._allowScalarOutput = false;
    this._isReduction = false;
    this._reduceDims = null;
    this._keepDim = false;
  }

  addInput(tensor) {
    this._inputs.push(tensor);
    return this;
  }

  addOutput(tensor) {
    this._output = tensor || null;
    return this;
  }

  setNumOutputs(n) {
    this._numOutputs = n;
    return this;
  }

  setEnforceDtype(dtype) {
    this._enforceDtype = dtype;
    return this;
  }

  setPromoteInputs(flag) {
    this._promoteInputs = flag;
    return this;
  }

  setCheckShapes(flag) {
    this._checkShapes = flag;
    return this;
  }

  setAllowScalarOutput(flag) {
    this._allowScalarOutput = flag;
    return this;
  }

  declareReduction(dims, keepDim) {
    this._isReduction = true;
    this._reduceDims = dims;
    this._keepDim = keepDim ?? false;
    return this;
  }

  build() {
    return new TensorIterator(this);
  }
}
