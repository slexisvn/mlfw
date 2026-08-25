import { describe, it, expect } from 'vitest';
import { printLIR, LIRPrinter } from '../../../../src/compiler/ir/lir/printer.js';
import { lowerToLIR } from '../../../../src/compiler/passes/lowering/tensor_to_lir.js';
import { schemaNodeTypes } from '../../../../src/compiler/ir/ir_visitor.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import {
  LIRFunc, LIRFlatLoadNode, LIRFlatStoreNode, LIRAccumulatorNode, LIRBindingsNode, LIRMetadata,
} from '../../../../src/compiler/ir/lir/nodes.js';
import {
  PrimFunc, ForNode, BlockNode, BlockRealizeNode, SeqNode,
  BufferStoreNode, BufferLoadNode, CallExternNode,
  MathOpNode, VariableNode, IntImmNode, FloatImmNode,
  ForKind, IterVarKind,
} from '../../../../src/compiler/ir/tensor/nodes.js';

function buf(name, shape, dtype = 'f32') {
  return new Buffer(name, shape, dtype, 'global');
}

function idx(name) {
  return new VariableNode(name, 'index');
}

function lines(node) {
  return printLIR(node).split('\n');
}

function flatOffset(base, stride, extra) {
  return new MathOpNode('+', new MathOpNode('*', idx(base), new IntImmNode(stride)), idx(extra));
}

describe('printLIR renders the LIR-only nodes', () => {
  it('prints a flat load as a single linear index into its buffer', () => {
    const x = buf('x', [4, 8]);
    const load = new LIRFlatLoadNode(x, flatOffset('i', 8, 'j'), 'f32');

    expect(printLIR(load)).toBe('x[((i * 8) + j)]');
  });

  it('prints a flat store with the value it writes', () => {
    const x = buf('x', [4]);
    const y = buf('y', [4]);
    const store = new LIRFlatStoreNode(y, idx('i'), new LIRFlatLoadNode(x, idx('i'), 'f32'), 'f32');

    expect(printLIR(store)).toBe('y[i] = x[i]');
  });

  it('prints the bindings of a block above the body they scope', () => {
    const y = buf('y', [4]);
    const bindings = new LIRBindingsNode(
      [{ name: 'v0', expr: idx('i') }, { name: 'v1', expr: new MathOpNode('+', idx('j'), new IntImmNode(1)) }],
      new LIRFlatStoreNode(y, idx('v0'), new FloatImmNode(0), 'f32'),
    );

    expect(lines(bindings)).toEqual([
      'bind v0 = i',
      'bind v1 = (j + 1)',
      'y[v0] = 0',
    ]);
  });

  it('prints a lir_func with its buffer map and body', () => {
    const x = buf('x', [4]);
    const y = buf('y', [4]);
    const argX = idx('arg_x');
    const argY = idx('arg_y');
    const body = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL,
      new LIRFlatStoreNode(y, idx('i'), new LIRFlatLoadNode(x, idx('i'), 'f32'), 'f32'));
    const func = new LIRFunc('copy', [argX, argY], body, new Map([[argX, x], [argY, y]]), [], new Map(), new LIRMetadata());

    expect(lines(func)).toEqual([
      'lir_func copy(arg_x, arg_y) {',
      '  x = buffer_map(arg_x, shape=[4], dtype=f32)',
      '  y = buffer_map(arg_y, shape=[4], dtype=f32)',
      '  for i in 0..4 {',
      '    y[i] = x[i]',
      '  }',
      '}',
    ]);
  });
});

describe('printLIR renders an accumulator as the loop it stands for', () => {
  const out = buf('out', [4]);
  const x = buf('x', [4, 8]);

  function accumulator(extra = {}) {
    return new LIRAccumulatorNode({
      localName: '_acc_0',
      dtype: 'f32',
      op: '+',
      initLoad: new LIRFlatLoadNode(out, idx('i'), 'f32'),
      loopVar: idx('k'),
      extent: new IntImmNode(8),
      loopKind: ForKind.SERIAL,
      body: new LIRFlatLoadNode(x, flatOffset('i', 8, 'k'), 'f32'),
      flushStore: new LIRFlatStoreNode(out, idx('i'), null, 'f32'),
      ...extra,
    });
  }

  it('prints the init load, the reduction loop and the flush store of the local', () => {
    expect(lines(accumulator())).toEqual([
      'accumulator _acc_0: f32 {',
      '  _acc_0 = out[i]',
      '  for k in 0..8 {',
      '    _acc_0 += x[((i * 8) + k)]',
      '  }',
      '  out[i] = _acc_0',
      '}',
    ]);
  });

  it('prints the accumulate operator of a non-additive reduction', () => {
    expect(lines(accumulator({ op: 'max' }))[3]).toBe('    _acc_0 max= x[((i * 8) + k)]');
  });

  it('annotates the reduction loop when it is not serial', () => {
    expect(lines(accumulator({ loopKind: ForKind.UNROLLED }))[2]).toBe('  for k in 0..8 @unrolled {');
  });

  it('prints the init body before the loop and the prologue inside it', () => {
    const initBody = new LIRFlatStoreNode(out, idx('i'), new FloatImmNode(0), 'f32');
    const prologue = new LIRFlatStoreNode(out, idx('k'), new FloatImmNode(1), 'f32');

    expect(lines(accumulator({ initBody, prologue }))).toEqual([
      'accumulator _acc_0: f32 {',
      '  _acc_0 = out[i]',
      '  out[i] = 0',
      '  for k in 0..8 {',
      '    out[k] = 1',
      '    _acc_0 += x[((i * 8) + k)]',
      '  }',
      '  out[i] = _acc_0',
      '}',
    ]);
  });

  it('marks a flush store printed outside its accumulator, whose value is the local', () => {
    const dangling = new LIRFlatStoreNode(out, idx('i'), null, 'f32');
    expect(printLIR(dangling)).toBe('out[i] = <acc>');
  });

  it('restores the enclosing local after a nested accumulator', () => {
    const inner = accumulator({ localName: '_acc_1' });
    const outer = accumulator({ body: inner });

    const printed = lines(outer);
    expect(printed.filter((line) => line.includes('_acc_1 ='))).toEqual(['      _acc_1 = out[i]']);
    expect(printed).toContain('      out[i] = _acc_1');
    expect(printed[printed.length - 2]).toBe('  out[i] = _acc_0');
  });
});

describe('printLIR keeps rendering the tensor IR nodes that survive lowering', () => {
  it('prints a lowered prim_func without reporting an unknown node', () => {
    const x = buf('x', [4, 8]);
    const y = buf('y', [4, 8]);
    const argX = idx('arg_x');
    const argY = idx('arg_y');
    const store = new BufferStoreNode(y, [idx('vi'), idx('vj')],
      new CallExternNode('tanh', [new BufferLoadNode(x, [idx('vi'), idx('vj')])], 'f32'));
    const block = new BlockNode(
      'tanh_block',
      [new BlockRealizeNode(idx('vi'), idx('i'), IterVarKind.DATA_PAR), new BlockRealizeNode(idx('vj'), idx('j'), IterVarKind.DATA_PAR)],
      [{ buffer: x }],
      [{ buffer: y }],
      store,
    );
    const inner = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.VECTORIZED, block);
    const outer = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.PARALLEL, inner);
    const primFunc = new PrimFunc('forward', [argX, argY], outer, new Map([[argX, x], [argY, y]]));

    const printed = printLIR(lowerToLIR(primFunc, CPUTarget()));

    expect(printed).not.toContain('UnknownNode');
    expect(printed.split('\n')).toEqual([
      'lir_func forward(arg_x, arg_y) {',
      '  x = buffer_map(arg_x, shape=[4,8], dtype=f32)',
      '  y = buffer_map(arg_y, shape=[4,8], dtype=f32)',
      '  for i in 0..4 @parallel {',
      '    for j in 0..8 @vectorized {',
      '      bind vi = i',
      '      bind vj = j',
      '      y[((vi * 8) + vj)] = tanh(x[((vi * 8) + vj)])',
      '    }',
      '  }',
      '}',
    ]);
  });

  it('prints a sequence of lowered statements one per line', () => {
    const y = buf('y', [2]);
    const seq = new SeqNode([
      new LIRFlatStoreNode(y, new IntImmNode(0), new FloatImmNode(1), 'f32'),
      new LIRFlatStoreNode(y, new IntImmNode(1), new FloatImmNode(2), 'f32'),
    ]);

    expect(lines(seq)).toEqual(['y[0] = 1', 'y[1] = 2']);
  });
});

describe('the printer covers every node type the traversal schema knows', () => {
  it('has a visit method for each schema node type, so nothing renders as UnknownNode', () => {
    const printer = new LIRPrinter();
    const missing = schemaNodeTypes().filter((type) => typeof printer['visit' + type] !== 'function');

    expect(missing, `node types the printer cannot render: ${missing.join(', ')}`).toEqual([]);
  });
});
