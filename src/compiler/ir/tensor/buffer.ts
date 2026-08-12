import { dtypeBytes } from '../../../util/dtype_map.js';
import { shapeProduct, symbolicShapeProduct, DYNAMIC } from '../graph/types.js';
import type { Dim, Shape } from '../graph/types.js';
import type { TensorNode, TirNode } from './nodes.js';

export class Buffer {
  name: string;
  shape: Shape;
  dtype: string;
  scope: string;
  offset: number;
  alignment: number;
  broadcastDims: readonly number[] | null;
  declare symbolicShape?: Shape;
  declare poolByteOffset?: number;
  declare align16?: boolean;
  declare storageAlign?: { axis: number; factor: number; offset: number };
  strides: Dim[];

  constructor(name: string, shape: Shape, dtype: string, scope: string, strides: Dim[] | null = null, offset = 0, alignment = 64) {
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
        if (typeof shape[i] === 'number') s *= shape[i] as number;
        else s = DYNAMIC;
      }
    }
  }

  get rank(): number {
    return this.shape.length;
  }

  get isScalar(): boolean {
    return this.shape.length === 0;
  }

  numel(): number {
    return shapeProduct(this.shape, -1);
  }

  symbolicNumel(): ReturnType<typeof symbolicShapeProduct> {
    return symbolicShapeProduct(this.shape);
  }

  sizeInBytes(): number {
    const n = this.numel();
    if (n < 0) return -1;
    return n * dtypeBytes(this.dtype);
  }
}

export type BufferRegionLike = { buffer: Buffer; min?: TirNode; extent?: TirNode };

export class BufferRegion {
  buffer: Buffer;
  min: TensorNode;
  extent: TensorNode;

  constructor(buffer: Buffer, min: TensorNode, extent: TensorNode) {
    this.buffer = buffer;
    this.min = min;
    this.extent = extent;
  }
}
