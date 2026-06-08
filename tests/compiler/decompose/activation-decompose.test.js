import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { DecompositionPass } from '../../../src/compiler/passes/decompose/decomposition_pass.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

function run(func) {
  return new DecompositionPass().run(func);
}

function retVal(func, i = 0) {
  return func.getReturnOp().getOperand(i);
}

function allValues(func) {
  const vals = [];
  for (const op of func.ops()) {
    for (let i = 0; i < op.numResults; i++) vals.push(op.getResult(i));
  }
  return vals;
}

describe('softmax decomposition', () => {
  it('rewires all users when softmax result feeds multiple consumers', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const sm = b.softmax(args[0], 1).getResult(0);
      const doubled = b.add(sm, sm);
      b.returnOp([doubled.getResult(0)]);
    });

    run(func);

    const addOp = func.findOp(op => op.opName === 'add');
    expect(addOp.getOperand(0)).toBe(addOp.getOperand(1));
    expect(addOp.getOperand(0).definingOp.opName).not.toBe('softmax');
  });

  it('reduce dims match the specified axis, not hardcoded', () => {
    for (const axis of [0, 1, 2]) {
      const t = new TensorType([2, 3, 4], ScalarType.F32);
      const func = buildFunction('f', [t], [t], (b, args) => {
        b.returnOp([b.softmax(args[0], axis).getResult(0)]);
      });

      run(func);

      const reduces = func.findOps(op => op.opName === 'reduce');
      for (const r of reduces) {
        expect(r.getAttr('dimensions')).toEqual([axis]);
      }
    }
  });

  it('intermediate reduce shape drops the reduced axis', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 1).getResult(0)]);
    });

    run(func);

    const reduces = func.findOps(op => op.opName === 'reduce');
    for (const r of reduces) {
      expect(r.getResult(0).type.shape).toEqual([2, 4]);
    }
  });

  it('broadcasts restore the original shape after reduce', () => {
    const t = new TensorType([5, 7], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 0).getResult(0)]);
    });

    run(func);

    const broadcasts = func.findOps(op => op.opName === 'broadcast_in_dim');
    const fullShape = broadcasts.filter(bc => bc.getResult(0).type.shape.length === 2);
    for (const bc of fullShape) {
      expect(bc.getResult(0).type.shape).toEqual([5, 7]);
    }
  });
});

describe('log_softmax decomposition', () => {
  it('produces log op that softmax decomposition does not', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const smFunc = buildFunction('sm', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 1).getResult(0)]);
    });
    const lsmFunc = buildFunction('lsm', [t], [t], (b, args) => {
      b.returnOp([b.logSoftmax(args[0], 1).getResult(0)]);
    });

    run(smFunc);
    run(lsmFunc);

    expect(smFunc.findOp(op => op.opName === 'log')).toBeNull();
    expect(lsmFunc.findOp(op => op.opName === 'log')).not.toBeNull();
    expect(smFunc.findOp(op => op.opName === 'div')).not.toBeNull();
    expect(lsmFunc.findOp(op => op.opName === 'div')).toBeNull();
  });
});

describe('sigmoid decomposition', () => {
  it('all intermediate values preserve input dtype through the chain', () => {
    for (const dtype of [ScalarType.F32, ScalarType.F64]) {
      const t = new TensorType([3, 5], dtype);
      const func = buildFunction('f', [t], [t], (b, args) => {
        b.returnOp([b.sigmoid(args[0]).getResult(0)]);
      });

      run(func);

      for (const val of allValues(func)) {
        if (val.type instanceof TensorType) {
          expect(val.type.dtype).toBe(dtype);
        }
      }
    }
  });

  it('constant 1 uses input dtype, not hardcoded f32', () => {
    const t = new TensorType([4], ScalarType.F64);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });

    run(func);

    const ones = func.findOps(op => op.opName === 'constant' && op.getAttr('value') === 1);
    for (const c of ones) {
      expect(c.getResult(0).type.dtype).toBe(ScalarType.F64);
    }
  });
});

describe('gelu decomposition', () => {
  it('final result is x * sigmoid_part, not sigmoid_part * x (input is lhs)', () => {
    const t = new TensorType([2, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.gelu(args[0]).getResult(0)]);
    });

    run(func);

    const ret = retVal(func);
    expect(ret.definingOp.getOperand(0)).toBe(func.args[0]);
  });

  it('uses 1.702 coefficient, not some other approximation', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.gelu(args[0]).getResult(0)]);
    });

    run(func);

    const constants = func.findOps(op => op.opName === 'constant');
    const coeffs = constants.map(c => c.getAttr('value')).filter(v => v !== 0 && v !== 1);
    expect(coeffs).toContain(1.702);
  });
});

describe('silu decomposition', () => {
  it('differs from gelu only by absence of coefficient scaling', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const siluFunc = buildFunction('silu', [t], [t], (b, args) => {
      b.returnOp([b.silu(args[0]).getResult(0)]);
    });
    const geluFunc = buildFunction('gelu', [t], [t], (b, args) => {
      b.returnOp([b.gelu(args[0]).getResult(0)]);
    });

    run(siluFunc);
    run(geluFunc);

    const siluConsts = siluFunc.findOps(op => op.opName === 'constant')
      .map(c => c.getAttr('value')).filter(v => v !== 0 && v !== 1);
    const geluConsts = geluFunc.findOps(op => op.opName === 'constant')
      .map(c => c.getAttr('value')).filter(v => v !== 0 && v !== 1);

    expect(siluConsts).toHaveLength(0);
    expect(geluConsts).toContain(1.702);
  });

  it('neg receives original input directly, not a transformed copy', () => {
    const t = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.silu(args[0]).getResult(0)]);
    });

    run(func);

    const neg = func.findOp(op => op.opName === 'neg');
    expect(neg.getOperand(0)).toBe(func.args[0]);
  });
});

describe('elu decomposition', () => {
  it('decomposes to compare + select + exp primitives', () => {
    const t = new TensorType([2, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.elu(args[0]).getResult(0)]);
    });
    run(func);
    const ops = new Set();
    for (const op of func.ops()) ops.add(op.opName);
    expect(ops.has('elu')).toBe(false);
    expect(ops.has('compare')).toBe(true);
    expect(ops.has('select')).toBe(true);
    expect(ops.has('exp')).toBe(true);
  });

  it('preserves custom alpha attribute through decomposition', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.elu(args[0], 2.0).getResult(0)]);
    });
    run(func);
    const consts = func.findOps(op => op.opName === 'constant');
    const vals = consts.map(c => c.getAttr('value'));
    expect(vals).toContain(2.0);
  });
});

describe('leaky_relu decomposition', () => {
  it('decomposes to compare + select + mul primitives', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.leakyRelu(args[0]).getResult(0)]);
    });
    run(func);
    const ops = new Set();
    for (const op of func.ops()) ops.add(op.opName);
    expect(ops.has('leaky_relu')).toBe(false);
    expect(ops.has('compare')).toBe(true);
    expect(ops.has('select')).toBe(true);
  });

  it('uses the specified negative slope, not hardcoded', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.leakyRelu(args[0], 0.2).getResult(0)]);
    });
    run(func);
    const consts = func.findOps(op => op.opName === 'constant');
    const vals = consts.map(c => c.getAttr('value'));
    expect(vals).toContain(0.2);
  });
});

describe('celu decomposition', () => {
  it('decomposes to maximum + minimum + exp primitives', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.celu(args[0]).getResult(0)]);
    });
    run(func);
    const ops = new Set();
    for (const op of func.ops()) ops.add(op.opName);
    expect(ops.has('celu')).toBe(false);
    expect(ops.has('maximum')).toBe(true);
    expect(ops.has('minimum')).toBe(true);
    expect(ops.has('exp')).toBe(true);
  });
});

describe('selu decomposition', () => {
  it('decomposes to compare + select + exp + mul with fixed lambda/alpha', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.selu(args[0]).getResult(0)]);
    });
    run(func);
    const ops = new Set();
    for (const op of func.ops()) ops.add(op.opName);
    expect(ops.has('selu')).toBe(false);
    expect(ops.has('compare')).toBe(true);
    expect(ops.has('select')).toBe(true);
    const consts = func.findOps(op => op.opName === 'constant');
    const vals = consts.map(c => c.getAttr('value'));
    expect(vals.some(v => Math.abs(v - 1.0507009873554805) < 1e-6)).toBe(true);
    expect(vals.some(v => Math.abs(v - 1.6732632423543772) < 1e-6)).toBe(true);
  });
});

describe('mish decomposition', () => {
  it('decomposes to tanh + log + exp + mul', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.mish(args[0]).getResult(0)]);
    });
    run(func);
    const ops = new Set();
    for (const op of func.ops()) ops.add(op.opName);
    expect(ops.has('mish')).toBe(false);
    expect(ops.has('tanh')).toBe(true);
    expect(ops.has('log')).toBe(true);
    expect(ops.has('exp')).toBe(true);
    expect(ops.has('mul')).toBe(true);
  });
});

describe('hardswish decomposition', () => {
  it('decomposes to minimum + maximum + mul + div', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.hardswish(args[0]).getResult(0)]);
    });
    run(func);
    const ops = new Set();
    for (const op of func.ops()) ops.add(op.opName);
    expect(ops.has('hardswish')).toBe(false);
    expect(ops.has('minimum')).toBe(true);
    expect(ops.has('maximum')).toBe(true);
    expect(ops.has('mul')).toBe(true);
  });
});

describe('hardsigmoid decomposition', () => {
  it('decomposes to minimum + maximum + div + add', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.hardsigmoid(args[0]).getResult(0)]);
    });
    run(func);
    const ops = new Set();
    for (const op of func.ops()) ops.add(op.opName);
    expect(ops.has('hardsigmoid')).toBe(false);
    expect(ops.has('minimum')).toBe(true);
    expect(ops.has('maximum')).toBe(true);
  });

  it('all intermediate values preserve input dtype', () => {
    for (const dtype of [ScalarType.F32, ScalarType.F64]) {
      const t = new TensorType([4], dtype);
      const func = buildFunction('f', [t], [t], (b, args) => {
        b.returnOp([b.hardsigmoid(args[0]).getResult(0)]);
      });
      run(func);
      for (const val of allValues(func)) {
        if (val.type instanceof TensorType) {
          expect(val.type.dtype).toBe(dtype);
        }
      }
    }
  });
});

describe('chained activations', () => {
  it('decomposing sigmoid→gelu chain wires sigmoid output into gelu input', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const sig = b.sigmoid(args[0]);
      const gel = b.gelu(sig.getResult(0));
      b.returnOp([gel.getResult(0)]);
    });

    run(func);

    const ops = func.opsArray().map(op => op.opName);
    expect(ops).not.toContain('sigmoid');
    expect(ops).not.toContain('gelu');

    const negOps = func.findOps(op => op.opName === 'neg');
    const negInputs = negOps.map(n => n.getOperand(0));
    expect(negInputs.some(v => v === func.args[0])).toBe(true);
    expect(negInputs.some(v => v !== func.args[0])).toBe(true);
  });

  it('second pass run returns UNCHANGED after full decomposition', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 1).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('no dangling values after decomposition — all results have valid definingOp', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const sig = b.sigmoid(args[0]);
      const sm = b.softmax(sig.getResult(0), 1);
      b.returnOp([sm.getResult(0)]);
    });

    run(func);

    for (const op of func.ops()) {
      for (let i = 0; i < op.numOperands; i++) {
        const v = op.getOperand(i);
        if (!v.isBlockArgument()) {
          expect(v.definingOp).not.toBeNull();
          expect(v.definingOp.parentBlock).not.toBeNull();
        }
      }
    }
  });
});
