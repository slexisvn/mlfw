import { Tensor } from '../../tensor/core/tensor.js';
import { fromBuffer } from '../../tensor/factory/from_ops.js';
import { typedArrayCtor } from '../../tensor/types/dtype.js';
import type { DType, NumericTypedArray } from '../../tensor/types/dtype.js';

const MAGIC = 'mlfw-ckpt-v1';
const HEADER_PREFIX_BYTES = 8;

type SerializableTypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array
  | BigInt64Array
  | BigUint64Array;

type SerializableTypedArrayConstructor =
  | Int8ArrayConstructor
  | Uint8ArrayConstructor
  | Uint8ClampedArrayConstructor
  | Int16ArrayConstructor
  | Uint16ArrayConstructor
  | Uint32ArrayConstructor
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | BigInt64ArrayConstructor
  | BigUint64ArrayConstructor;

type BufferMeta =
  | { kind: 'tensor'; dtype: DType; shape: readonly number[]; offset?: number; length?: number }
  | { kind: 'array'; arrayType: keyof typeof TYPED_ARRAY_CTORS; offset?: number; length?: number };

type BufferRecord = {
  meta: BufferMeta;
  bytes: Uint8Array;
};

type EncodedNode =
  | null
  | string
  | number
  | boolean
  | { $buf: number }
  | { $map: Array<[unknown, EncodedNode]> }
  | EncodedNode[]
  | { [key: string]: EncodedNode };

type BufferNode = { $buf: number };
type MapNode = { $map: Array<[unknown, EncodedNode]> };

const TYPED_ARRAY_CTORS = {
  Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array,
  Int32Array, Uint32Array, Float32Array, Float64Array,
  BigInt64Array, BigUint64Array,
};

function isTypedArray(value: unknown): value is SerializableTypedArray {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function byteView(arr: SerializableTypedArray | NumericTypedArray): Uint8Array {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

function typedArrayName(value: SerializableTypedArray): keyof typeof TYPED_ARRAY_CTORS {
  return value.constructor.name as keyof typeof TYPED_ARRAY_CTORS;
}

function isBufferNode(node: object): node is BufferNode {
  return '$buf' in node && typeof (node as { $buf?: unknown }).$buf === 'number';
}

function isMapNode(node: object): node is MapNode {
  return '$map' in node && Array.isArray((node as { $map?: unknown }).$map);
}

export function serializeCheckpoint(checkpoint: unknown): Uint8Array {
  const buffers: BufferRecord[] = [];
  const tree = encodeNode(checkpoint, buffers);

  let offset = 0;
  for (const buf of buffers) {
    buf.meta.offset = offset;
    buf.meta.length = buf.bytes.length;
    offset += buf.bytes.length;
  }

  const header = { format: MAGIC, tree, buffers: buffers.map((b) => b.meta) };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));

  const out = new Uint8Array(HEADER_PREFIX_BYTES + headerBytes.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  out.set(headerBytes, HEADER_PREFIX_BYTES);

  let pos = HEADER_PREFIX_BYTES + headerBytes.length;
  for (const buf of buffers) {
    out.set(buf.bytes, pos);
    pos += buf.bytes.length;
  }
  return out;
}

export function deserializeCheckpoint(bytes: Uint8Array | ArrayBufferLike): unknown {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const dataStart = HEADER_PREFIX_BYTES + headerLength;
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(HEADER_PREFIX_BYTES, dataStart))) as {
    format: string;
    tree: EncodedNode;
    buffers: BufferMeta[];
  };
  if (header.format !== MAGIC) {
    throw new Error('mlfw: unrecognized checkpoint format: ' + header.format);
  }

  const decoded = header.buffers.map((meta) => decodeBuffer(meta, u8, dataStart));
  return decodeNode(header.tree, decoded);
}

function encodeNode(value: unknown, buffers: BufferRecord[]): EncodedNode {
  if (value instanceof Tensor) {
    return pushBuffer(buffers, { kind: 'tensor', dtype: value.dtype, shape: value.shape }, byteView(value._impl.storage.data!));
  }
  if (isTypedArray(value)) {
    return pushBuffer(buffers, { kind: 'array', arrayType: typedArrayName(value) }, byteView(value));
  }
  if (value instanceof Map) {
    const entries: Array<[unknown, EncodedNode]> = [];
    for (const [k, v] of value) entries.push([k, encodeNode(v, buffers)]);
    return { $map: entries };
  }
  if (Array.isArray(value)) {
    return value.map((v) => encodeNode(v, buffers));
  }
  if (value && typeof value === 'object') {
    const obj: { [key: string]: EncodedNode } = {};
    const record = value as Record<string, unknown>;
    for (const key of Object.keys(record)) obj[key] = encodeNode(record[key], buffers);
    return obj;
  }
  return value as EncodedNode;
}

function pushBuffer(buffers: BufferRecord[], meta: BufferMeta, bytes: Uint8Array): { $buf: number } {
  const idx = buffers.length;
  buffers.push({ meta, bytes });
  return { $buf: idx };
}

function decodeBuffer(meta: BufferMeta, u8: Uint8Array, dataStart: number): Tensor | SerializableTypedArray {
  const start = dataStart + meta.offset!;
  const slice = u8.subarray(start, start + meta.length!);
  const Ctor = meta.kind === 'tensor' ? typedArrayCtor(meta.dtype) : TYPED_ARRAY_CTORS[meta.arrayType];
  const arrayType = meta.kind === 'tensor' ? undefined : meta.arrayType;
  if (!Ctor) throw new Error('mlfw: unknown typed array in checkpoint: ' + arrayType);
  const arr = new (Ctor as SerializableTypedArrayConstructor)(meta.length! / Ctor.BYTES_PER_ELEMENT);
  byteView(arr).set(slice);
  return meta.kind === 'tensor' ? fromBuffer(arr as NumericTypedArray, meta.shape, meta.dtype) : arr;
}

function decodeNode(node: EncodedNode, decoded: Array<Tensor | SerializableTypedArray>): unknown {
  if (node === null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map((n) => decodeNode(n, decoded));
  if (isBufferNode(node)) return decoded[node.$buf];
  if (isMapNode(node)) {
    const map = new Map();
    for (const [k, v] of node.$map) map.set(k, decodeNode(v, decoded));
    return map;
  }
  const obj: Record<string, unknown> = {};
  for (const key of Object.keys(node)) obj[key] = decodeNode(node[key], decoded);
  return obj;
}
