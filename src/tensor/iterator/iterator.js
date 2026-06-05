import { broadcastShapesMulti, computeStrides, computeNumel, isContiguous } from '../utils/shape_utils.js';
import { broadcastStrides } from '../utils/shape_utils.js';
import { resultDtype, typedArrayCtor, dtypeSize } from '../types/dtype.js';
import { empty } from '../factory/creation_ops.js';

export class TensorIterator {
  constructor(config) {
    this._inputs = config._inputs;
    this._isReduction = config._isReduction;
    this._reduceDims = config._reduceDims;
    this._keepDim = config._keepDim;
    this._output = config._output;
    this._numOutputs = config._numOutputs;

    this._dtype = this._resolveDtype(config);
    this._shape = this._resolveShape(config);
    this._numel = computeNumel(this._shape);

    this._allStrides = this._resolveStrides();
    this._allData = [];
    this._allOffsets = [];

    this._outputTensor = this._resolveOutput(config);

    this._collectDataPtrs();
    this._contiguous = this._checkContiguous();
  }

  _resolveDtype(config) {
    if (config._enforceDtype) return config._enforceDtype;
    if (this._inputs.length === 0) return 'f32';
    if (!config._promoteInputs) return this._inputs[0].dtype;

    let dt = this._inputs[0].dtype;
    for (let i = 1; i < this._inputs.length; i++) {
      dt = resultDtype(dt, this._inputs[i].dtype);
    }
    return dt;
  }

  _resolveShape(config) {
    if (this._isReduction) {
      return this._resolveReductionShape();
    }

    const shapes = this._inputs.map(t => t.shape);
    if (shapes.length === 0) return [];
    if (shapes.length === 1) return [...shapes[0]];

    const result = broadcastShapesMulti(shapes);
    if (!result) {
      const shapeStrs = shapes.map(s => `[${s.join(',')}]`).join(', ');
      throw new Error(`Cannot broadcast shapes: ${shapeStrs}`);
    }
    return result;
  }

  _resolveReductionShape() {
    if (this._inputs.length === 0) return [];
    const inputShape = this._inputs[0].shape;
    if (!this._reduceDims) return [];

    const dims = new Set(
      this._reduceDims.map(d => d < 0 ? inputShape.length + d : d)
    );

    return inputShape;
  }

  _resolveStrides() {
    const strides = [];

    if (this._output) {
      strides.push(broadcastStrides(this._output.shape, this._output.strides, this._shape));
    } else {
      strides.push(computeStrides(this._shape));
    }

    for (const inp of this._inputs) {
      strides.push(broadcastStrides(inp.shape, inp.strides, this._shape));
    }

    return strides;
  }

  _resolveOutput(config) {
    if (config._output) return config._output;

    if (this._isReduction && this._reduceDims) {
      const inputShape = this._inputs.length > 0 ? this._inputs[0].shape : [];
      const dims = new Set(
        this._reduceDims.map(d => d < 0 ? inputShape.length + d : d)
      );
      const outShape = [];
      for (let i = 0; i < inputShape.length; i++) {
        if (dims.has(i)) {
          if (this._keepDim) outShape.push(1);
        } else {
          outShape.push(inputShape[i]);
        }
      }
      return empty(outShape, {
        dtype: this._dtype,
        device: this._inputs.length > 0 ? this._inputs[0].device : undefined,
      });
    }

    return empty(this._shape, {
      dtype: this._dtype,
      device: this._inputs.length > 0 ? this._inputs[0].device : undefined,
    });
  }

  _collectDataPtrs() {
    if (this._outputTensor) {
      const s = this._outputTensor._impl.storage;
      this._allData.push(s ? s.data : null);
      this._allOffsets.push(this._outputTensor._impl.storageOffset);
    }

    for (const inp of this._inputs) {
      const s = inp._impl.storage;
      this._allData.push(s ? s.data : null);
      this._allOffsets.push(inp._impl.storageOffset);
    }
  }

  _checkContiguous() {
    const refStrides = computeStrides(this._shape);
    for (let t = 0; t < this._allStrides.length; t++) {
      const s = this._allStrides[t];
      for (let d = 0; d < this._shape.length; d++) {
        if (this._shape[d] > 1 && s[d] !== refStrides[d]) return false;
      }
    }
    return true;
  }

  get shape() { return this._shape; }
  get numel() { return this._numel; }
  get ndim() { return this._shape.length; }
  get dtype() { return this._dtype; }
  get isContiguous() { return this._contiguous; }
  get numInputs() { return this._inputs.length; }
  get numTensors() { return this._allData.length; }
  get isReduction() { return this._isReduction; }
  get reduceDims() { return this._reduceDims; }
  get keepDim() { return this._keepDim; }

  output() { return this._outputTensor; }

  data(i) { return this._allData[i]; }
  stridesAt(i) { return this._allStrides[i]; }
  offsetAt(i) { return this._allOffsets[i]; }
  inputShape() { return this._inputs.length > 0 ? this._inputs[0].shape : this._shape; }
}
