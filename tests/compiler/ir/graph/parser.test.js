import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../../src/compiler/ir/graph/module.js';
import { printModule, printFunction } from '../../../../src/compiler/ir/graph/printer.js';
import { parseModule, parseFunction, IRParseError } from '../../../../src/compiler/ir/graph/parser.js';
import { verifyModule, verifyFunction } from '../../../../src/compiler/ir/graph/verifier.js';
import { TensorType, ScalarType, Layout, DYNAMIC } from '../../../../src/compiler/ir/graph/types.js';
import { SymInt } from '../../../../src/compiler/ir/sym_int.js';
import { tensor } from '../../../../src/index.js';
import { trace } from '../../../../src/tracing/compile.js';
import * as nn from '../../../../src/nn/index.js';
import * as T from '../../../../src/tensor/ops/ops.js';
import { mulberry32 } from '../../../_utils/rng.js';
import { randomNested } from '../../../_utils/tensor_data.js';
import { PassManager } from '../../../../src/compiler/passes/pass_manager.js';
import { buildGraphPipeline } from '../../../../src/compiler/pipeline/graph_pipeline.js';
import { CompilerConfig } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';

const rng = mulberry32(9091);
const grid = (shape) => randomNested(rng, shape);

function f32(shape) {
  return new TensorType(shape, ScalarType.F32);
}

function moduleOf(...funcs) {
  const module = new GraphModule('m');
  for (const func of funcs) module.addFunction(func);
  return module;
}

function roundTrip(module) {
  const once = printModule(module);
  const reparsed = parseModule(once);
  return { once, twice: printModule(reparsed), reparsed };
}

function expectStable(module) {
  const { once, twice, reparsed } = roundTrip(module);
  expect(twice).toBe(once);
  return reparsed;
}

describe('printed IR parses back to an equivalent module', () => {
  it('round-trips a straight-line elementwise function', () => {
    const t = f32([4, 8]);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const s = b.add(args[0], args[1]);
      const m = b.mul(s.getResult(0), args[0]);
      b.returnOp([b.tanh(m.getResult(0)).getResult(0)]);
    });

    const reparsed = expectStable(moduleOf(func));
    const parsed = reparsed.getFunction('f');
    expect(parsed.opsArray().map(o => o.opName)).toEqual(['add', 'mul', 'tanh', 'return']);
    expect(verifyFunction(parsed).map(e => e.message)).toEqual([]);
  });

  it('rebuilds the use-def edges, not just the op sequence', () => {
    const t = f32([4]);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const s = b.add(args[0], args[1]);
      b.returnOp([b.mul(s.getResult(0), s.getResult(0)).getResult(0)]);
    });

    const parsed = expectStable(moduleOf(func)).getFunction('f');
    const mul = parsed.findOp(o => o.opName === 'mul');
    const add = parsed.findOp(o => o.opName === 'add');
    expect(mul.getOperand(0)).toBe(add.getResult(0));
    expect(mul.getOperand(1)).toBe(add.getResult(0));
    expect(add.getResult(0).useCount).toBe(2);
  });

  it('round-trips several functions in one module', () => {
    const t = f32([2]);
    const a = buildFunction('first', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const c = buildFunction('second', [t, t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[1]).getResult(0)]);
    });

    const reparsed = expectStable(moduleOf(a, c));
    expect(reparsed.functionNames()).toEqual(['first', 'second']);
    expect(reparsed.name).toBe('m');
  });
});

describe('the harder parts of the grammar survive the trip', () => {
  it('round-trips region-carrying control flow with block arguments', () => {
    const t = f32([4]);
    const boolT = new TensorType([], ScalarType.BOOL);
    const func = buildFunction('branchy', [t, t, boolT], [t], (b, args) => {
      const branch = b.ifOp(args[2], [t], (tb) => {
        tb.yieldOp([tb.add(args[0], args[1]).getResult(0)]);
      }, (eb) => {
        eb.yieldOp([eb.mul(args[0], args[1]).getResult(0)]);
      });
      b.returnOp([branch.getResult(0)]);
    });

    const parsed = expectStable(moduleOf(func)).getFunction('branchy');
    const ifOp = parsed.findOp(o => o.opName === 'if');
    expect(ifOp.numRegions).toBe(2);
    expect(ifOp.getRegion(0).entryBlock.opsArray().map(o => o.opName)).toEqual(['add', 'yield']);
    expect(ifOp.getRegion(1).entryBlock.opsArray().map(o => o.opName)).toEqual(['mul', 'yield']);
  });

  it('round-trips a scan region whose body reads its own block arguments', () => {
    const xs = f32([3, 4]);
    const carry = f32([4]);
    const func = buildFunction('scanner', [xs, carry], [carry], (b, args) => {
      const scan = b.scanOp([args[0]], [args[1]], (bb, xt, c) => {
        return [[bb.add(xt[0], c[0]).getResult(0)], []];
      });
      b.returnOp([scan.getResult(0)]);
    });

    const parsed = expectStable(moduleOf(func)).getFunction('scanner');
    const scan = parsed.findOp(o => o.opName === 'scan');
    const body = scan.getRegion(0).entryBlock;
    expect(body.arguments.length).toBe(2);
    const add = body.opsArray().find(o => o.opName === 'add');
    expect(add.getOperand(0)).toBe(body.arguments[0]);
    expect(add.getOperand(1)).toBe(body.arguments[1]);
  });

  it('round-trips a region body that captures a value from the enclosing block', () => {
    const t = f32([4]);
    const boolT = new TensorType([], ScalarType.BOOL);
    const func = buildFunction('capture', [t, boolT], [t], (b, args) => {
      const outer = b.neg(args[0]);
      const branch = b.ifOp(args[1], [t], (tb) => {
        tb.yieldOp([tb.add(outer.getResult(0), outer.getResult(0)).getResult(0)]);
      }, (eb) => {
        eb.yieldOp([outer.getResult(0)]);
      });
      b.returnOp([branch.getResult(0)]);
    });

    const parsed = expectStable(moduleOf(func)).getFunction('capture');
    const neg = parsed.findOp(o => o.opName === 'neg');
    const inner = parsed.findOp(o => o.opName === 'if').getRegion(0).entryBlock.opsArray()[0];
    expect(inner.getOperand(0)).toBe(neg.getResult(0));
  });

  it('round-trips dense constant data without collapsing it to a string', () => {
    const t = f32([2, 2]);
    const data = Float32Array.from([1.5, -2.25, 0, 8]);
    const func = buildFunction('weights', [t], [t], (b, args) => {
      const w = b.constant(data, t);
      b.returnOp([b.add(args[0], w.getResult(0)).getResult(0)]);
    });

    const parsed = expectStable(moduleOf(func)).getFunction('weights');
    const value = parsed.findOp(o => o.opName === 'constant').getAttr('value');
    expect(value).toBeInstanceOf(Float32Array);
    expect([...value]).toEqual([1.5, -2.25, 0, 8]);
  });

  it('round-trips dynamic and symbolic dimensions', () => {
    const n = SymInt.var('n');
    const symbolic = new TensorType([n, 8], ScalarType.F32);
    const dynamic = new TensorType([DYNAMIC, 8], ScalarType.F32);
    const func = buildFunction('shapes', [symbolic, dynamic], [symbolic], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const parsed = expectStable(moduleOf(func)).getFunction('shapes');
    expect(parsed.inputTypes[0].shape[0]).toBeInstanceOf(SymInt);
    expect(parsed.inputTypes[0].shape[0].name).toBe('n');
    expect(parsed.inputTypes[1].shape[0]).toBe(DYNAMIC);
  });

  it('round-trips a compound symbolic dimension', () => {
    const t = new TensorType([SymInt.mul(SymInt.var('n'), 4), 8], ScalarType.F32);
    const func = buildFunction('sym', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const parsed = expectStable(moduleOf(func)).getFunction('sym');
    expect(SymInt.equals(parsed.inputTypes[0].shape[0], t.shape[0])).toBe(true);
  });

  it('round-trips a non-identity layout', () => {
    const t = new TensorType([4, 8], ScalarType.F32, new Layout([1, 0]));
    const func = buildFunction('laid_out', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const parsed = expectStable(moduleOf(func)).getFunction('laid_out');
    expect(parsed.inputTypes[0].layout.order).toEqual([1, 0]);
  });

  it('round-trips array, string, boolean and infinite attribute values', () => {
    const t = f32([2, 3]);
    const out = f32([3, 2]);
    const func = buildFunction('attrs', [t], [out], (b, args) => {
      const tr = b.transpose(args[0], [1, 0]);
      b.returnOp([tr.getResult(0)]);
    });
    const transpose = func.findOp(o => o.opName === 'transpose');
    transpose.setAttr('label', 'a name with spaces');
    transpose.setAttr('flag', true);
    transpose.setAttr('threshold', -Infinity);
    transpose.setAttr('nested', [[1, 2], [3, 4]]);

    const parsed = expectStable(moduleOf(func)).getFunction('attrs');
    const reparsed = parsed.findOp(o => o.opName === 'transpose');
    expect(reparsed.getAttr('label')).toBe('a name with spaces');
    expect(reparsed.getAttr('flag')).toBe(true);
    expect(reparsed.getAttr('threshold')).toBe(-Infinity);
    expect(reparsed.getAttr('nested')).toEqual([[1, 2], [3, 4]]);
    expect(reparsed.getAttr('permutation')).toEqual([1, 0]);
  });

  it('round-trips IR whose textual order is not topological', () => {
    const t = f32([4]);
    const func = buildFunction('reordered', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const m = b.mul(a.getResult(0), args[0]);
      b.returnOp([m.getResult(0)]);
    });

    const block = func.entryBlock;
    const add = func.findOp(o => o.opName === 'add');
    const mul = func.findOp(o => o.opName === 'mul');
    block.removeOp(add);
    block.insertAfter(add, mul);

    const printed = printFunction(func);
    expect(printed.indexOf('mul')).toBeLessThan(printed.indexOf('add'));

    const parsed = parseFunction(printed);
    expect(printFunction(parsed)).toBe(printed);
    expect(parsed.findOp(o => o.opName === 'mul').getOperand(0))
      .toBe(parsed.findOp(o => o.opName === 'add').getResult(0));
  });
});

describe('a parsed module is real IR, not a transcript', () => {
  it('parses text written by hand and verifies it', () => {
    const text = [
      'module @hand {',
      '  func @f(%0: tensor<4xf32>, %1: tensor<4xf32>) -> (tensor<4xf32>) {',
      '    %2 = add(%0, %1) : tensor<4xf32>',
      '    %3 = mul(%2, %0) : tensor<4xf32>',
      '    return(%3)',
      '  }',
      '}',
    ].join('\n');

    const module = parseModule(text);
    expect(verifyModule(module).map(e => e.message)).toEqual([]);
    expect(printModule(module)).toBe(text);
  });

  it('parses a scalar (rank 0) tensor type', () => {
    const text = [
      'module @scalar {',
      '  func @f(%0: tensor<f32>) -> (tensor<f32>) {',
      '    return(%0)',
      '  }',
      '}',
    ].join('\n');

    const module = parseModule(text);
    expect(module.getFunction('f').inputTypes[0].rank).toBe(0);
    expect(printModule(module)).toBe(text);
  });

  it('normalizes the legacy rank-0 spelling to the canonical one', () => {
    const module = parseModule([
      'module @scalar {',
      '  func @f(%0: tensor<xf32>) -> (tensor<xf32>) {',
      '    return(%0)',
      '  }',
      '}',
    ].join('\n'));

    expect(module.getFunction('f').inputTypes[0].rank).toBe(0);
    expect(printModule(module)).toContain('tensor<f32>');
    expect(printModule(module)).not.toContain('tensor<xf32>');
  });

  it('rejects an empty dimension in a ranked tensor type', () => {
    const text = [
      'module @bad {',
      '  func @f(%0: tensor<x4xf32>) -> (tensor<x4xf32>) {',
      '    return(%0)',
      '  }',
      '}',
    ].join('\n');

    expect(() => parseModule(text)).toThrowError(/invalid dimension/);
  });
});

describe('parse errors point at the offending line', () => {
  it('rejects a use of an undefined value', () => {
    const text = [
      'module @bad {',
      '  func @f(%0: tensor<4xf32>) -> (tensor<4xf32>) {',
      '    %1 = add(%0, %9) : tensor<4xf32>',
      '    return(%1)',
      '  }',
      '}',
    ].join('\n');

    expect(() => parseModule(text)).toThrowError(IRParseError);
    expect(() => parseModule(text)).toThrowError(/line 3.*undefined value '%9'/);
  });

  it('rejects a result list that disagrees with the declared types', () => {
    const text = [
      'module @bad {',
      '  func @f(%0: tensor<4xf32>) -> (tensor<4xf32>) {',
      '    %1, %2 = add(%0, %0) : tensor<4xf32>',
      '    return(%1)',
      '  }',
      '}',
    ].join('\n');

    expect(() => parseModule(text)).toThrowError(/line 3.*names 2 results but declares 1 result types/);
  });

  it('rejects a tensor type that does not end in a known dtype', () => {
    const text = [
      'module @bad {',
      '  func @f(%0: tensor<4xquux>) -> (tensor<4xquux>) {',
      '    return(%0)',
      '  }',
      '}',
    ].join('\n');

    expect(() => parseModule(text)).toThrowError(/line 2.*'tensor<4xquux>' does not end in a known dtype/);
  });

  it('rejects an unterminated region', () => {
    const text = [
      'module @bad {',
      '  func @f(%0: tensor<xbool>) -> () {',
      '    if(%0)',
      '    {',
      '      yield()',
      '  }',
      '}',
    ].join('\n');

    expect(() => parseModule(text)).toThrowError(IRParseError);
  });
});

describe('IR produced by tracing real models round-trips', () => {
  const MODELS = [
    {
      name: 'MLP with GELU',
      build: () => {
        const layers = [new nn.Linear(8, 16), new nn.Linear(16, 4)];
        return {
          fwd: (x) => layers[1].forward(nn.F.gelu(layers[0].forward(x))),
          inputs: [tensor(grid([2, 8]))],
        };
      },
    },
    {
      name: 'CNN with pooling and flatten',
      build: () => {
        const conv = new nn.Conv2d(1, 4, 3, { padding: 1 });
        const pool = new nn.MaxPool2d(2);
        const flat = new nn.Flatten();
        const fc = new nn.Linear(4 * 2 * 2, 3);
        return {
          fwd: (x) => fc.forward(flat.forward(pool.forward(nn.F.relu(conv.forward(x))))),
          inputs: [tensor(grid([1, 1, 4, 4]))],
        };
      },
    },
    {
      name: 'LayerNorm + softmax attention block',
      build: () => {
        const norm = new nn.LayerNorm(8);
        const proj = new nn.Linear(8, 8);
        return {
          fwd: (x) => T.softmax(proj.forward(norm.forward(x)), 1),
          inputs: [tensor(grid([2, 8]))],
        };
      },
    },
    {
      name: 'LSTM (scan region)',
      build: () => {
        const lstm = new nn.LSTM(4, 6);
        return {
          fwd: (x, h, c) => lstm.forward(x, [h, c])[0],
          inputs: [tensor(grid([2, 3, 4])), tensor(grid([1, 3, 6])), tensor(grid([1, 3, 6]))],
        };
      },
    },
  ];

  for (const model of MODELS) {
    it(`round-trips the traced graph of ${model.name}`, async () => {
      const { fwd, inputs } = model.build();
      const traced = await trace(fwd, inputs);
      const module = traced.graph ?? traced;

      const printed = printModule(module);
      const reparsed = parseModule(printed);

      expect(printModule(reparsed)).toBe(printed);
      expect(verifyModule(reparsed).map(e => e.message)).toEqual([]);

      const originalOps = [...module.functions()].flatMap(f => [...f.opsRecursive()].map(o => o.opName));
      const parsedOps = [...reparsed.functions()].flatMap(f => [...f.opsRecursive()].map(o => o.opName));
      expect(parsedOps).toEqual(originalOps);
      expect(originalOps.length).toBeGreaterThan(3);
    });
  }
});

describe('IR after the graph pipeline has rewritten it round-trips', () => {
  it('round-trips a module carrying materialized fusion regions', async () => {
    const layers = [new nn.Linear(8, 16), new nn.Linear(16, 8)];
    const fwd = (x) => T.mul(nn.F.gelu(layers[0].forward(x)), layers[0].forward(x));
    const traced = await trace((x) => layers[1].forward(fwd(x)), [tensor(grid([4, 8]))]);
    const module = traced.graph ?? traced;

    const pm = new PassManager();
    for (const pass of buildGraphPipeline(new CompilerConfig({ target: CPUTarget() }), CPUTarget())) {
      pm.addPass(pass);
    }
    pm.run(module);

    const fusions = [...module.functions()].flatMap(f => f.findOps(o => o.opName === 'fusion'));
    expect(fusions.length).toBeGreaterThan(0);
    expect(fusions.every(o => o.numRegions === 1)).toBe(true);

    const printed = printModule(module);
    const reparsed = parseModule(printed);
    expect(printModule(reparsed)).toBe(printed);
    expect(verifyModule(reparsed).map(e => e.message)).toEqual([]);

    const reparsedFusions = [...reparsed.functions()].flatMap(f => f.findOps(o => o.opName === 'fusion'));
    expect(reparsedFusions.length).toBe(fusions.length);
    for (let i = 0; i < fusions.length; i++) {
      const original = fusions[i].getRegion(0).entryBlock;
      const copy = reparsedFusions[i].getRegion(0).entryBlock;
      expect(copy.arguments.length).toBe(original.arguments.length);
      expect(copy.opsArray().map(o => o.opName)).toEqual(original.opsArray().map(o => o.opName));
    }
  });
});
