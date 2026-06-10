import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import {
  IfThenElseNode, WhileNode, BufferStoreNode, BufferLoadNode
} from '../../../src/compiler/ir/tensor/nodes.js';

const SKIP_KEYS = new Set(['_parent', '_parentKey', '_parentIdx']);

function collectNodes(node, predicate) {
  const result = [];
  const visited = new Set();
  const stack = [node];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object' || visited.has(n)) continue;
    visited.add(n);
    if (predicate(n)) result.push(n);
    for (const key of Object.keys(n)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = n[key];
      if (Array.isArray(val)) for (const item of val) stack.push(item);
      else if (val && typeof val === 'object') stack.push(val);
    }
  }
  return result;
}

const F32 = ScalarType.F32;
const BOOL = ScalarType.BOOL;

describe('if lowering result unification', () => {
  it('both then and else branches write into the same result buffer', () => {
    const t = new TensorType([4], F32);
    const pred = new TensorType([], BOOL);
    const func = buildFunction('f', [pred, t], [t], (b, args) => {
      const ifop = b.ifOp(args[0], [t],
        (tb) => { tb.yieldOp([tb.neg(args[1]).getResult(0)]); },
        (eb) => { eb.yieldOp([eb.exp(args[1]).getResult(0)]); });
      b.returnOp([ifop.getResult(0)]);
    });
    const pf = lowerGraphToPrimFunc(func);

    const ifNodes = collectNodes(pf.body, n => n instanceof IfThenElseNode);
    expect(ifNodes.length).toBe(1);
    const ifn = ifNodes[0];

    expect(ifn.elseBody).not.toBeNull();

    const thenStores = collectNodes(ifn.thenBody, n => n instanceof BufferStoreNode);
    const elseStores = collectNodes(ifn.elseBody, n => n instanceof BufferStoreNode);
    const thenTargets = new Set(thenStores.map(s => s.buffer.name));
    const elseTargets = new Set(elseStores.map(s => s.buffer.name));

    const shared = [...thenTargets].filter(n => elseTargets.has(n));
    expect(shared.length).toBeGreaterThan(0);
  });
});

describe('while lowering loop carry and condition', () => {
  it('stores predicate into condVar and copies body yields back into loop buffers', () => {
    const t = new TensorType([4], F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const wop = b.whileOp([args[0]],
        (cb, cargs) => {
          const anyOp = cb.scalarConstant(1, BOOL);
          cb.yieldOp([anyOp.getResult(0)]);
        },
        (bb, bargs) => {
          bb.yieldOp([bb.neg(bargs[0]).getResult(0)]);
        });
      b.returnOp([wop.getResult(0)]);
    });
    const pf = lowerGraphToPrimFunc(func);

    const whileNodes = collectNodes(pf.body, n => n instanceof WhileNode);
    expect(whileNodes.length).toBe(1);
    const wn = whileNodes[0];

    expect(wn.condVar).toBeDefined();
    expect(wn.condVar.shape).toEqual([]);

    const condStores = collectNodes(wn.condBody, n => n instanceof BufferStoreNode);
    const storesIntoCond = condStores.filter(s => s.buffer === wn.condVar);
    expect(storesIntoCond.length).toBe(1);
    expect(storesIntoCond[0].value).toBeInstanceOf(BufferLoadNode);

    const bodyStores = collectNodes(wn.loopBody, n => n instanceof BufferStoreNode);
    expect(bodyStores.length).toBeGreaterThan(0);
  });
});
