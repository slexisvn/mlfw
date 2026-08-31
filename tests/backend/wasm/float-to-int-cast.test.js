import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { WasmTarget, CPUTarget } from '../../../src/compiler/support/target.js';

const F32 = ScalarType.F32;
const I32 = ScalarType.I32;

describe('WASM float-to-int cast saturates instead of trapping', () => {
  const HOSTILE = [NaN, Infinity, -Infinity, 1e30, -1e30, 2147483648, -2147483904, 3.7, -3.7, 0];

  function castKernel(name) {
    return buildFunction(name, [new TensorType([HOSTILE.length], F32)], [new TensorType([HOSTILE.length], I32)],
      (b, args) => { b.returnOp([b.convert(args[0], I32).getResult(0)]); });
  }

  it('does not trap on NaN, infinities or out-of-range magnitudes', () => {
    const input = Float32Array.from(HOSTILE);
    const out = new Int32Array(HOSTILE.length);
    const result = compileGraph(castKernel('sat_cast'), WasmTarget(), { scheduling: { enabled: true } });
    expect(() => result.run('sat_cast', input, out)).not.toThrow();
  });

  it('emits the non-trapping opcode', () => {
    const source = compileGraph(castKernel('sat_src'), WasmTarget()).getSource('sat_src');
    expect(source).toContain('i32.trunc_sat_f32_s');
    expect(source).not.toContain('i32.trunc_f32_s\n');
  });

  it('saturates to the i32 bounds and maps NaN to zero', () => {
    const input = Float32Array.from(HOSTILE);
    const out = new Int32Array(HOSTILE.length);
    compileGraph(castKernel('sat_values'), WasmTarget()).run('sat_values', input, out);

    expect(out[0], 'NaN').toBe(0);
    expect(out[1], '+Infinity').toBe(2147483647);
    expect(out[2], '-Infinity').toBe(-2147483648);
    expect(out[3], '1e30').toBe(2147483647);
    expect(out[4], '-1e30').toBe(-2147483648);
    expect(out[7], '3.7 truncates toward zero').toBe(3);
    expect(out[8], '-3.7 truncates toward zero').toBe(-3);
    expect(out[9], 'zero').toBe(0);
  });

  it('agrees with the CPU backend on in-range values', () => {
    const values = Float32Array.from([0, 1, -1, 3.7, -3.7, 127, -128, 1000.9, -1000.9]);
    const t = new TensorType([values.length], F32);
    const mk = (name) => buildFunction(name, [t], [new TensorType([values.length], I32)],
      (b, args) => { b.returnOp([b.convert(args[0], I32).getResult(0)]); });

    const wasmOut = new Int32Array(values.length);
    const cpuOut = new Int32Array(values.length);
    compileGraph(mk('cw'), WasmTarget()).run('cw', values, wasmOut);
    compileGraph(mk('cc'), CPUTarget()).run('cc', values, cpuOut);
    expect([...wasmOut]).toEqual([...cpuOut]);
  });
});
