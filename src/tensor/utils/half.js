const _f32buf = new Float32Array(1);
const _u32buf = new Uint32Array(_f32buf.buffer);

const F16_ADD_BIAS = ((((15 - 127) << 23) >>> 0) + 0xfff) >>> 0;
const F16_MAGIC_MUL = 5.192296858534828e+33;

export function f16ToF32(h) {
  _u32buf[0] = (h & 0x7fff) << 13;
  _f32buf[0] *= F16_MAGIC_MUL;
  if (_f32buf[0] >= 65536) _u32buf[0] |= 0x7f800000;
  _u32buf[0] |= (h & 0x8000) << 16;
  return _f32buf[0];
}

export function f32ToF16(value) {
  _f32buf[0] = value;
  const sign = _u32buf[0] & 0x80000000;
  let x = (_u32buf[0] ^ sign) >>> 0;
  let o;
  if (x >= 0x47800000) {
    o = x > 0x7f800000 ? 0x7e00 : 0x7c00;
  } else if (x < 0x38800000) {
    _u32buf[0] = x;
    _f32buf[0] += 0.5;
    o = (_u32buf[0] - 0x3f000000) & 0xffff;
  } else {
    const mantOdd = (x >> 13) & 1;
    x = (x + F16_ADD_BIAS) >>> 0;
    x = (x + mantOdd) >>> 0;
    o = (x >> 13) & 0xffff;
  }
  return (o | (sign >>> 16)) & 0xffff;
}

export function bf16ToF32(h) {
  _u32buf[0] = (h << 16) >>> 0;
  return _f32buf[0];
}

export function f32ToBf16(value) {
  _f32buf[0] = value;
  const x = _u32buf[0];
  if (((x >>> 23) & 0xff) === 0xff) return (x >>> 16) & 0xffff;
  const rounding = (0x7fff + ((x >>> 16) & 1)) >>> 0;
  return (((x + rounding) >>> 0) >>> 16) & 0xffff;
}

export function coerceForStorage(dtype, v) {
  if (dtype === 'f16') return f32ToF16(v);
  if (dtype === 'bf16') return f32ToBf16(v);
  if (dtype === 'i64') return typeof v === 'bigint' ? v : BigInt(Math.trunc(v));
  return v;
}

export function readFromStorage(dtype, raw) {
  if (dtype === 'f16') return f16ToF32(raw);
  if (dtype === 'bf16') return bf16ToF32(raw);
  if (dtype === 'i64') return typeof raw === 'bigint' ? Number(raw) : raw;
  return raw;
}

export const HALF_WASM_CONSTANTS = {
  F16_ADD_BIAS: F16_ADD_BIAS | 0,
  F16_MAGIC_MUL,
};

const _g = globalThis;
_g.__mlfw_f16_to_f32 = f16ToF32;
_g.__mlfw_f32_to_f16 = f32ToF16;
_g.__mlfw_bf16_to_f32 = bf16ToF32;
_g.__mlfw_f32_to_bf16 = f32ToBf16;
