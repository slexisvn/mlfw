import { describe, it, expect } from 'vitest';
import { encodeWat } from '../../../src/backend/wasm/wat_encoder.js';

function instantiate(wat, name) {
  const inst = new WebAssembly.Instance(new WebAssembly.Module(encodeWat(wat)), {});
  return { fn: inst.exports[name], mem: inst.exports.memory };
}

describe('wat_encoder f64x2 SIMD + named params', () => {
  it('f64x2 add/mul with named params round-trips two lanes', () => {
    const wat = `(module (memory (export "memory") 1 10)
      (func (export "vop") (param $a i32) (param $b i32) (param $out i32) (local $va v128) (local $vb v128)
        (local.get $a) v128.load local.set $va
        (local.get $b) v128.load local.set $vb
        (local.get $out)
        (local.get $va) (local.get $vb) f64x2.add
        (local.get $va) (local.get $vb) f64x2.mul
        f64x2.sub v128.store))`;
    const { fn, mem } = instantiate(wat, 'vop');
    const a = new Float64Array(mem.buffer, 0, 2);
    const b = new Float64Array(mem.buffer, 16, 2);
    const out = new Float64Array(mem.buffer, 32, 2);
    a[0] = 3; a[1] = 5; b[0] = 2; b[1] = 4;
    fn(0, 16, 32);
    expect(out[0]).toBeCloseTo(3 + 2 - 3 * 2, 12);
    expect(out[1]).toBeCloseTo(5 + 4 - 5 * 4, 12);
  });

  it('f64x2 splat/sqrt/extract_lane compute per-lane', () => {
    const wat = `(module (memory (export "memory") 1 10)
      (func (export "g") (param $out i32) (local $v v128)
        (f64.const 16) f64x2.splat f64x2.sqrt local.set $v
        (local.get $out) (local.get $v) f64x2.extract_lane 0 f64.store
        (local.get $out) (i32.const 8) i32.add (local.get $v) f64x2.extract_lane 1 f64.store))`;
    const { fn, mem } = instantiate(wat, 'g');
    fn(0);
    const out = new Float64Array(mem.buffer, 0, 2);
    expect(out[0]).toBeCloseTo(4, 12);
    expect(out[1]).toBeCloseTo(4, 12);
  });

  it('named params resolve to the correct local indices', () => {
    const wat = `(module (memory (export "memory") 1 10)
      (func (export "h") (param $p i32) (param $q i32) (param $out i32)
        (local.get $out)
        (local.get $p) f64.load (local.get $q) f64.load f64.sub
        f64.store))`;
    const { fn, mem } = instantiate(wat, 'h');
    const p = new Float64Array(mem.buffer, 0, 1);
    const q = new Float64Array(mem.buffer, 8, 1);
    const out = new Float64Array(mem.buffer, 16, 1);
    p[0] = 10; q[0] = 3;
    fn(0, 8, 16);
    expect(out[0]).toBeCloseTo(7, 12);
  });
});
