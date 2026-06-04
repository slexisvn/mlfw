import { Layout, DYNAMIC } from './types.js';

export class BlockedLayout {
  constructor(baseOrder, blockFactors = new Map()) {
    this.baseOrder = Object.freeze([...baseOrder]);
    this.blockFactors = new Map(blockFactors);
    this._hash = null;
  }

  static fromLayout(layout) {
    return new BlockedLayout(layout.order);
  }

  get logicalRank() {
    return this.baseOrder.length;
  }

  get physicalRank() {
    return this.baseOrder.length + this.blockFactors.size;
  }

  isBlocked() {
    return this.blockFactors.size > 0;
  }

  isIdentity() {
    if (this.blockFactors.size > 0) return false;
    for (let i = 0; i < this.baseOrder.length; i++) {
      if (this.baseOrder[i] !== i) return false;
    }
    return true;
  }

  getPhysicalShape(logicalShape) {
    const physShape = [];
    const blockEntries = this._sortedBlockEntries();
    let blockIdx = 0;

    for (let i = 0; i < this.baseOrder.length; i++) {
      const dim = this.baseOrder[i];
      const blockSize = this.blockFactors.get(dim);
      if (blockSize !== undefined) {
        const logDim = logicalShape[dim];
        if (logDim === DYNAMIC) {
          physShape.push(DYNAMIC);
        } else {
          physShape.push(Math.ceil(logDim / blockSize));
        }
      } else {
        physShape.push(logicalShape[dim]);
      }
    }

    for (const [dim, blockSize] of blockEntries) {
      physShape.push(blockSize);
    }

    return physShape;
  }

  computeStrides(shape) {
    const physShape = this.getPhysicalShape(shape);
    const n = physShape.length;
    const strides = new Array(n);
    let stride = 1;
    for (let i = n - 1; i >= 0; i--) {
      strides[i] = stride;
      if (physShape[i] === DYNAMIC) {
        stride = DYNAMIC;
      } else if (stride !== DYNAMIC) {
        stride *= physShape[i];
      }
    }
    return strides;
  }

  mapLogicalToPhysical(logicalIndices, exprBuilder) {
    if (!this.isBlocked()) {
      const result = new Array(this.baseOrder.length);
      for (let i = 0; i < this.baseOrder.length; i++) {
        result[i] = logicalIndices[this.baseOrder[i]];
      }
      return result;
    }

    const physIndices = [];
    const blockEntries = this._sortedBlockEntries();

    for (let i = 0; i < this.baseOrder.length; i++) {
      const dim = this.baseOrder[i];
      const blockSize = this.blockFactors.get(dim);
      if (blockSize !== undefined) {
        physIndices.push(exprBuilder.floorDiv(logicalIndices[dim], blockSize));
      } else {
        physIndices.push(logicalIndices[dim]);
      }
    }

    for (const [dim, blockSize] of blockEntries) {
      physIndices.push(exprBuilder.mod(logicalIndices[dim], blockSize));
    }

    return physIndices;
  }

  toLayout() {
    if (this.isBlocked()) {
      throw new Error('Cannot convert blocked layout to plain Layout');
    }
    return new Layout(this.baseOrder);
  }

  compose(other) {
    if (this.isBlocked() || other.isBlocked()) {
      throw new Error('Cannot compose blocked layouts');
    }
    const result = new Array(this.baseOrder.length);
    for (let i = 0; i < this.baseOrder.length; i++) {
      result[i] = this.baseOrder[other.baseOrder[i]];
    }
    return new BlockedLayout(result);
  }

  inverse() {
    if (this.isBlocked()) {
      throw new Error('Cannot invert blocked layout directly');
    }
    const inv = new Array(this.baseOrder.length);
    for (let i = 0; i < this.baseOrder.length; i++) {
      inv[this.baseOrder[i]] = i;
    }
    return new BlockedLayout(inv);
  }

  equals(other) {
    if (this === other) return true;
    if (!(other instanceof BlockedLayout)) return false;
    if (this.baseOrder.length !== other.baseOrder.length) return false;
    for (let i = 0; i < this.baseOrder.length; i++) {
      if (this.baseOrder[i] !== other.baseOrder[i]) return false;
    }
    if (this.blockFactors.size !== other.blockFactors.size) return false;
    for (const [dim, size] of this.blockFactors) {
      if (other.blockFactors.get(dim) !== size) return false;
    }
    return true;
  }

  hash() {
    if (this._hash !== null) return this._hash;
    let h = 0x811c9dc5;
    for (let i = 0; i < this.baseOrder.length; i++) {
      h = ((h ^ this.baseOrder[i]) * 0x01000193) & 0x7fffffff;
    }
    for (const [dim, size] of this._sortedBlockEntries()) {
      h = ((h ^ (dim << 16 | size)) * 0x01000193) & 0x7fffffff;
    }
    this._hash = h;
    return h;
  }

  _sortedBlockEntries() {
    return [...this.blockFactors.entries()].sort((a, b) => a[0] - b[0]);
  }
}
