import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType, DYNAMIC, typeToString } from '../../../src/compiler/ir/graph/types.js';
import { GraphFunction } from '../../../src/compiler/ir/graph/function.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { IRBuilder, buildFunction, buildModule } from '../../../src/compiler/ir/graph/builder.js';
import { verifyModule, verifyFunction } from '../../../src/compiler/ir/verifier/verifier.js';
import { printModule, printFunction } from '../../../src/compiler/ir/printer/printer.js';
import { registry } from '../../../src/compiler/ir/graph/ops.js';

const f32 = ScalarType.F32;
const f32_2x3 = new TensorType([2, 3], f32);
const f32_3x4 = new TensorType([3, 4], f32);
const f32_2x4 = new TensorType([2, 4], f32);

describe('IRBuilder elementwise ops', () => {
  it('builds add with broadcast', () => {
    const func = buildFunction('test_add', [f32_2x3, f32_2x3], [f32_2x3], (b, [x, y]) => {
      const add = b.add(x, y);
      b.returnOp([add.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
    assert.equal(func.numOps(), 2);
  });

  it('builds chain: add -> mul -> neg', () => {
    const func = buildFunction('test_chain', [f32_2x3, f32_2x3], [f32_2x3], (b, [x, y]) => {
      const add = b.add(x, y);
      const mul = b.mul(add.getResult(0), x);
      const neg = b.neg(mul.getResult(0));
      b.returnOp([neg.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
  });

  it('builds unary math ops', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('test_unary', [f32_4], [f32_4], (b, [x]) => {
      const e = b.exp(x);
      const l = b.log(e.getResult(0));
      const s = b.sqrt(l.getResult(0));
      const t = b.tanh(s.getResult(0));
      b.returnOp([t.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
  });
});

describe('IRBuilder matmul', () => {
  it('builds 2D matmul', () => {
    const func = buildFunction('test_matmul', [f32_2x3, f32_3x4], [f32_2x4], (b, [x, y]) => {
      const dot = b.matmul(x, y);
      b.returnOp([dot.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
    const dotOp = func.findOp(op => op.opName === 'dot');
    assert.ok(dotOp);
    const resType = dotOp.getResult(0).type;
    assert.deepEqual([...resType.shape], [2, 4]);
  });
});

describe('IRBuilder reduce', () => {
  it('builds sum reduction', () => {
    const func = buildFunction('test_reduce', [f32_2x3], [new TensorType([2], f32)], (b, [x]) => {
      const zero = b.scalarConstant(0, f32);
      const red = b.reduce(x, zero.getResult(0), [1], 'sum');
      b.returnOp([red.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
    const redOp = func.findOp(op => op.opName === 'reduce');
    assert.deepEqual([...redOp.getResult(0).type.shape], [2]);
  });
});

describe('IRBuilder reshape/transpose', () => {
  it('builds reshape', () => {
    const func = buildFunction('test_reshape', [f32_2x3], [new TensorType([6], f32)], (b, [x]) => {
      const r = b.reshape(x, [6]);
      b.returnOp([r.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
  });

  it('builds transpose', () => {
    const func = buildFunction('test_transpose', [f32_2x3], [new TensorType([3, 2], f32)], (b, [x]) => {
      const t = b.transpose(x, [1, 0]);
      b.returnOp([t.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
  });
});

describe('IRBuilder broadcast', () => {
  it('builds broadcast_in_dim', () => {
    const scalar = new TensorType([], f32);
    const func = buildFunction('test_bcast', [scalar], [f32_2x3], (b, [x]) => {
      const bc = b.broadcast(x, [2, 3], []);
      b.returnOp([bc.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
  });
});

describe('IRBuilder compare/select', () => {
  it('builds compare + select', () => {
    const boolType = new TensorType([4], ScalarType.BOOL);
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('test_select', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      const cmp = b.compare(x, y, 'gt');
      const sel = b.select(cmp.getResult(0), x, y);
      b.returnOp([sel.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
  });
});

describe('IRBuilder relu', () => {
  it('builds relu pattern', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('test_relu', [f32_4], [f32_4], (b, [x]) => {
      const r = b.relu(x);
      b.returnOp([r.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
  });
});

describe('IRBuilder dynamic shapes', () => {
  it('builds ops with dynamic shapes', () => {
    const dynType = new TensorType([DYNAMIC, 3], f32);
    const func = buildFunction('test_dynamic', [dynType, dynType], [dynType], (b, [x, y]) => {
      const add = b.add(x, y);
      b.returnOp([add.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
    const addOp = func.findOp(op => op.opName === 'add');
    assert.ok(addOp.getResult(0).type.hasDynamic);
  });
});

describe('IRBuilder conv', () => {
  it('builds conv2d', () => {
    const input = new TensorType([1, 3, 32, 32], f32);
    const kernel = new TensorType([16, 3, 3, 3], f32);
    const output = new TensorType([1, 16, 32, 32], f32);
    const func = buildFunction('test_conv', [input, kernel], [output], (b, [x, w]) => {
      const c = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
      b.returnOp([c.getResult(0)]);
    });
    const errors = verifyFunction(func);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
  });
});

describe('Module and printer', () => {
  it('prints a module', () => {
    const mod = buildModule('test_module', [
      ['add_func', [f32_2x3, f32_2x3], [f32_2x3], (b, [x, y]) => {
        const add = b.add(x, y);
        b.returnOp([add.getResult(0)]);
      }]
    ]);
    const errors = verifyModule(mod);
    assert.equal(errors.length, 0, errors.map(e => e.toString()).join('\n'));
    const text = printModule(mod);
    assert.ok(text.includes('module @test_module'));
    assert.ok(text.includes('func @add_func'));
    assert.ok(text.includes('add'));
    assert.ok(text.includes('return'));
  });

  it('prints deterministic value names', () => {
    const func = buildFunction('f', [f32_2x3], [f32_2x3], (b, [x]) => {
      const neg = b.neg(x);
      b.returnOp([neg.getResult(0)]);
    });
    const text = printFunction(func);
    assert.ok(text.includes('%0'));
    assert.ok(text.includes('%1'));
  });
});

describe('Verifier catches errors', () => {
  it('detects missing return', () => {
    const func = new GraphFunction('bad', [f32_2x3], [f32_2x3]);
    const errors = verifyFunction(func);
    assert.ok(errors.some(e => e.message.includes('return')));
  });

  it('detects output count mismatch', () => {
    const func = buildFunction('bad2', [f32_2x3], [f32_2x3, f32_2x3], (b, [x]) => {
      b.returnOp([x]);
    });
    const errors = verifyFunction(func);
    assert.ok(errors.some(e => e.message.includes('operands')));
  });
});

describe('Operation structural hash', () => {
  it('equal ops have same hash', () => {
    const func = buildFunction('hash_test', [f32_2x3, f32_2x3], [f32_2x3], (b, [x, y]) => {
      const a1 = b.add(x, y);
      const a2 = b.add(x, y);
      b.returnOp([a1.getResult(0)]);
    });
    const ops = func.findOps(op => op.opName === 'add');
    assert.equal(ops.length, 2);
    assert.equal(ops[0].structuralHash(), ops[1].structuralHash());
    assert.ok(ops[0].structuralEquals(ops[1]));
  });
});

describe('Operation erase', () => {
  it('erases unused op', () => {
    const func = buildFunction('erase_test', [f32_2x3], [f32_2x3], (b, [x]) => {
      const unused = b.neg(x);
      b.returnOp([x]);
    });
    const negOp = func.findOp(op => op.opName === 'neg');
    assert.ok(negOp);
    negOp.erase();
    assert.equal(func.findOp(op => op.opName === 'neg'), null);
  });

  it('throws when erasing op with used results', () => {
    const func = buildFunction('erase_fail', [f32_2x3], [f32_2x3], (b, [x]) => {
      const neg = b.neg(x);
      b.returnOp([neg.getResult(0)]);
    });
    const negOp = func.findOp(op => op.opName === 'neg');
    assert.throws(() => negOp.erase(), /still has uses/);
  });
});

describe('Op registry', () => {
  it('has all standard ops', () => {
    const expected = ['add', 'sub', 'mul', 'div', 'neg', 'exp', 'log', 'tanh',
                      'reduce', 'dot', 'reshape', 'transpose', 'conv', 'constant',
                      'broadcast_in_dim', 'compare', 'select', 'return', 'fusion',
                      'if', 'while', 'custom_call', 'concat', 'slice', 'pad',
                      'gather', 'scatter'];
    for (const name of expected) {
      assert.ok(registry.has(name), `Missing op: ${name}`);
    }
  });

  it('op traits are correct', () => {
    const addDef = registry.get('add');
    assert.ok(addDef.isCommutative);
    assert.ok(addDef.isElementwise);
    assert.ok(!addDef.hasSideEffects);
    const reduceDef = registry.get('reduce');
    assert.ok(reduceDef.isReduction);
  });
});
