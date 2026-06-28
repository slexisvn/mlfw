import { dtypeBytes } from '../../../backend/dtype_map.js';
import { shapeProduct, symbolicShapeProduct, DYNAMIC } from '../graph/types.js';

export class Buffer {
  constructor(name, shape, dtype, scope, strides = null, offset = 0, alignment = 64) {
    this.name = name;
    this.shape = shape;
    this.dtype = dtype;
    this.scope = scope;
    this.offset = offset;
    this.alignment = alignment;

    this.broadcastDims = null;

    if (strides) {
      this.strides = strides;
    } else {
      this.strides = new Array(shape.length);
      let s = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        this.strides[i] = s;
        if (s === DYNAMIC) continue;
        if (typeof shape[i] === 'number') s *= shape[i];
        else s = DYNAMIC;
      }
    }
  }

  get rank() {
    return this.shape.length;
  }

  get isScalar() {
    return this.shape.length === 0;
  }

  numel() {
    return shapeProduct(this.shape, -1);
  }

  symbolicNumel() {
    return symbolicShapeProduct(this.shape);
  }

  sizeInBytes() {
    const n = this.numel();
    if (n < 0) return -1;
    return n * dtypeBytes(this.dtype);
  }
}

export class BufferRegion {
  constructor(buffer, min, extent) {
    this.buffer = buffer;
    this.min = min;
    this.extent = extent;
  }
}
