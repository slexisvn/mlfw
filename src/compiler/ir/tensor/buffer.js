const DTYPE_BYTES = {
  'f16': 2, 'f32': 4, 'f64': 8,
  'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
  'bool': 1, 'index': 4
};

export class Buffer {
  constructor(name, shape, dtype, scope, strides = null, offset = 0, alignment = 64) {
    this.name = name;
    this.shape = shape;
    this.dtype = dtype;
    this.scope = scope;
    this.offset = offset;
    this.alignment = alignment;

    if (strides) {
      this.strides = strides;
    } else {
      this.strides = new Array(shape.length);
      let s = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        this.strides[i] = s;
        if (typeof shape[i] === 'number') s *= shape[i];
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
    let n = 1;
    for (let i = 0; i < this.shape.length; i++) {
      if (typeof this.shape[i] !== 'number') return -1;
      n *= this.shape[i];
    }
    return n;
  }

  sizeInBytes() {
    const n = this.numel();
    if (n < 0) return -1;
    return n * (DTYPE_BYTES[this.dtype] || 4);
  }
}

export class BufferRegion {
  constructor(buffer, min, extent) {
    this.buffer = buffer;
    this.min = min;
    this.extent = extent;
  }
}
