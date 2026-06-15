import { Tensor } from '../../tensor/core/tensor.js';
import { fromBuffer } from '../../tensor/factory/from_ops.js';
import { typedArrayCtor } from '../../tensor/types/dtype.js';

const MAGIC = 'mlfw-ckpt-v1';
const HEADER_PREFIX_BYTES = 8;

const TYPED_ARRAY_CTORS = {
  Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array,
  Int32Array, Uint32Array, Float32Array, Float64Array,
  BigInt64Array, BigUint64Array,
};

function isTypedArray(value) {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function byteView(arr) {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

export function serializeCheckpoint(checkpoint) {
  const buffers = [];
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

export function deserializeCheckpoint(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const dataStart = HEADER_PREFIX_BYTES + headerLength;
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(HEADER_PREFIX_BYTES, dataStart)));
  if (header.format !== MAGIC) {
    throw new Error('mlfw: unrecognized checkpoint format: ' + header.format);
  }

  const decoded = header.buffers.map((meta) => decodeBuffer(meta, u8, dataStart));
  return decodeNode(header.tree, decoded);
}

function encodeNode(value, buffers) {
  if (value instanceof Tensor) {
    return pushBuffer(buffers, { kind: 'tensor', dtype: value.dtype, shape: value.shape }, byteView(value._impl.storage.data));
  }
  if (isTypedArray(value)) {
    return pushBuffer(buffers, { kind: 'array', arrayType: value.constructor.name }, byteView(value));
  }
  if (value instanceof Map) {
    const entries = [];
    for (const [k, v] of value) entries.push([k, encodeNode(v, buffers)]);
    return { $map: entries };
  }
  if (Array.isArray(value)) {
    return value.map((v) => encodeNode(v, buffers));
  }
  if (value && typeof value === 'object') {
    const obj = {};
    for (const key of Object.keys(value)) obj[key] = encodeNode(value[key], buffers);
    return obj;
  }
  return value;
}

function pushBuffer(buffers, meta, bytes) {
  const idx = buffers.length;
  buffers.push({ meta, bytes });
  return { $buf: idx };
}

function decodeBuffer(meta, u8, dataStart) {
  const start = dataStart + meta.offset;
  const slice = u8.subarray(start, start + meta.length);
  const Ctor = meta.kind === 'tensor' ? typedArrayCtor(meta.dtype) : TYPED_ARRAY_CTORS[meta.arrayType];
  if (!Ctor) throw new Error('mlfw: unknown typed array in checkpoint: ' + meta.arrayType);
  const arr = new Ctor(meta.length / Ctor.BYTES_PER_ELEMENT);
  byteView(arr).set(slice);
  return meta.kind === 'tensor' ? fromBuffer(arr, meta.shape, meta.dtype) : arr;
}

function decodeNode(node, decoded) {
  if (node === null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map((n) => decodeNode(n, decoded));
  if ('$buf' in node) return decoded[node.$buf];
  if ('$map' in node) {
    const map = new Map();
    for (const [k, v] of node.$map) map.set(k, decodeNode(v, decoded));
    return map;
  }
  const obj = {};
  for (const key of Object.keys(node)) obj[key] = decodeNode(node[key], decoded);
  return obj;
}
