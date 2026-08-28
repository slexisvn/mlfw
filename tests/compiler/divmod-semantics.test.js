import { describe, it, expect } from 'vitest';
import { floorDiv, floorMod, truncDiv, truncMod } from '../../src/util/divmod.js';
import { SymInt } from '../../src/compiler/analysis/sym_int.js';
import { CPUCodegen } from '../../src/backend/cpu/codegen.js';
import { WasmCodegen } from '../../src/backend/wasm/codegen.js';
import { CUDACodegen } from '../../src/backend/cuda/codegen.js';
import { WebGPUCodegen } from '../../src/backend/webgpu/codegen.js';
import { CPUTarget, WasmTarget, CUDATarget, WebGPUTarget } from '../../src/backend/target.js';
import { encodeWat } from '../../src/backend/wasm/wat_encoder.js';
import { Buffer } from '../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  MathOpNode, VariableNode, IntImmNode, ForKind, mathOp,
} from '../../src/compiler/ir/tensor/nodes.js';

const DIVIDENDS = [-7, -6, -5, -1, 0, 1, 5, 6, 7];
const DIVISORS = [3, -3, 2, -2, 1, -1];

function divModKernel(dtype, divisor) {
  const x = new Buffer('x', [DIVIDENDS.length], dtype, 'global');
  const q = new Buffer('q', [DIVIDENDS.length], dtype, 'global');
  const r = new Buffer('r', [DIVIDENDS.length], dtype, 'global');
  const i = new VariableNode('i', 'index');
  const body = new SeqNode([
    new BufferStoreNode(q, [i], new MathOpNode('//', new BufferLoadNode(x, [i]), new IntImmNode(divisor))),
    new BufferStoreNode(r, [i], new MathOpNode('%', new BufferLoadNode(x, [i]), new IntImmNode(divisor))),
  ]);
  const blk = new BlockNode('divmod', [], [{ buffer: x }], [{ buffer: q }, { buffer: r }], body);
  const loop = new ForNode(i, new IntImmNode(0), new IntImmNode(DIVIDENDS.length), ForKind.SERIAL, blk);
  return new PrimFunc('divmod', ['p0', 'p1', 'p2'], loop, new Map([['p0', x], ['p1', q], ['p2', r]]));
}

function runOnCpu(divisor) {
  const src = new CPUCodegen(CPUTarget()).generate(divModKernel('i32', divisor));
  const fn = new Function('return ' + src)();
  const x = Int32Array.from(DIVIDENDS);
  const q = new Int32Array(DIVIDENDS.length);
  const r = new Int32Array(DIVIDENDS.length);
  fn(x, q, r);
  return { q: Array.from(q), r: Array.from(r) };
}

function runOnWasm(divisor) {
  const res = new WasmCodegen(WasmTarget()).generate(divModKernel('i32', divisor));
  const instance = new WebAssembly.Instance(new WebAssembly.Module(encodeWat(res.wat)), { math: {} });
  const memory = instance.exports.memory;
  const offsets = res.params.map((name) => res.bufferOffsets.get(name));
  new Int32Array(memory.buffer, offsets[0], DIVIDENDS.length).set(DIVIDENDS);
  instance.exports.divmod(...offsets);
  return {
    q: Array.from(new Int32Array(memory.buffer, offsets[1], DIVIDENDS.length)),
    r: Array.from(new Int32Array(memory.buffer, offsets[2], DIVIDENDS.length)),
  };
}

function evaluateEmittedExpr(source, declarations = '') {
  const fn = new Function('a', 'b', declarations + 'return ' + source);
  return (a, b) => fn(a, b) | 0;
}

function floorHelpersAsJs(kernelSource) {
  const decls = [...kernelSource.matchAll(/(\w+)\((?:int a, int b|a: i32, b: i32)\)[^{]*\{ return (.+?); \}/g)];
  return decls.map(([, name, body]) => `function ${name}(a, b) { return (${body}) | 0; }`).join('');
}

const symbolicOperands = (op) => new MathOpNode(op, new VariableNode('a', 'index'), new VariableNode('b', 'index'));

describe('integer // and % mean floor division and floor modulo in every layer', () => {
  it('the scalar definitions satisfy a === (a // b) * b + (a % b) for every sign combination', () => {
    for (const a of DIVIDENDS) {
      for (const b of DIVISORS) {
        expect(floorDiv(a, b) * b + floorMod(a, b), `floor ${a} ${b}`).toBe(a);
        expect(truncDiv(a, b) * b + truncMod(a, b), `trunc ${a} ${b}`).toBe(a);
      }
    }
    expect(floorDiv(-3, 2)).toBe(-2);
    expect(floorMod(-3, 2)).toBe(1);
    expect(truncDiv(-3, 2)).toBe(-1);
    expect(truncMod(-3, 2)).toBe(-1);
  });

  it('TIR constant folding agrees with the symbolic shape layer on negative dividends', () => {
    for (const a of DIVIDENDS) {
      for (const b of DIVISORS) {
        const q = mathOp('//', new IntImmNode(a), new IntImmNode(b));
        const r = mathOp('%', new IntImmNode(a), new IntImmNode(b));
        expect(q.value, `// ${a} ${b}`).toBe(SymInt.div(a, b));
        expect(r.value, `% ${a} ${b}`).toBe(SymInt.mod(a, b));
      }
    }
  });

  it('the CPU backend computes floor // and % on negative dividends', () => {
    for (const b of DIVISORS) {
      const { q, r } = runOnCpu(b);
      expect(q, `quotients for divisor ${b}`).toEqual(DIVIDENDS.map((a) => floorDiv(a, b)));
      expect(r, `remainders for divisor ${b}`).toEqual(DIVIDENDS.map((a) => floorMod(a, b)));
    }
  });

  it('the WASM backend produces the same values as the CPU backend on negative dividends', () => {
    for (const b of DIVISORS) {
      const cpu = runOnCpu(b);
      const wasm = runOnWasm(b);
      expect(wasm.q, `quotients for divisor ${b}`).toEqual(cpu.q);
      expect(wasm.r, `remainders for divisor ${b}`).toEqual(cpu.r);
    }
  });

  it('the CUDA and WebGPU backends emit expressions that evaluate to the same floor results', () => {
    const cuda = new CUDACodegen(CUDATarget());
    const webgpu = new WebGPUCodegen(WebGPUTarget());
    const cudaHelpers = floorHelpersAsJs(new CUDACodegen(CUDATarget()).generate(divModKernel('i32', 3)).source);
    const wgslHelpers = floorHelpersAsJs(new WebGPUCodegen(WebGPUTarget()).generate(divModKernel('i32', 3)).source);
    for (const helpers of [cudaHelpers, wgslHelpers]) {
      expect(helpers).toContain('function floordiv');
      expect(helpers).toContain('function floormod');
    }
    const emitters = {
      cuda: {
        div: evaluateEmittedExpr(cuda._exprToC(symbolicOperands('//')), cudaHelpers),
        mod: evaluateEmittedExpr(cuda._exprToC(symbolicOperands('%')), cudaHelpers),
      },
      webgpu: {
        div: evaluateEmittedExpr(webgpu._exprToWGSL(symbolicOperands('//')), wgslHelpers),
        mod: evaluateEmittedExpr(webgpu._exprToWGSL(symbolicOperands('%')), wgslHelpers),
      },
    };

    for (const [name, emitted] of Object.entries(emitters)) {
      for (const a of DIVIDENDS) {
        for (const b of DIVISORS) {
          expect(emitted.div(a, b), `${name} ${a} // ${b}`).toBe(floorDiv(a, b));
          expect(emitted.mod(a, b), `${name} ${a} % ${b}`).toBe(floorMod(a, b));
        }
      }
    }
  });
});
