import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { BackwardGraphBuilder } from '../../../src/compiler/ad/backward_builder.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import '../../../src/compiler/ad/index.js';
import { F32 as F, T as t } from '../../_utils/ir_fixture.js';

const numel = (s) => s.reduce((a, b) => a * b, 1);
const T = 4;

function ifInScan(name) {
  return buildFunction(name, [t([T]), t([T])], [t([])], (b, [x, pred]) => {
    const init = b.scalarConstant(0, 'f32').getResult(0);
    const scan = b.scanOp([x, pred], [init], (bb, [xt, pt], [acc]) => {
      const zero = bb.scalarConstant(0, 'f32').getResult(0);
      const cmp = bb.compare(pt, zero, 'gt').getResult(0);
      const res = bb.ifOp(cmp, [t([])],
        (tb) => { tb.yieldOp([tb.add(acc, tb.mul(xt, xt).getResult(0)).getResult(0)]); },
        (eb) => { eb.yieldOp([eb.add(acc, xt).getResult(0)]); }).getResult(0);
      return [[res], []];
    });
    b.returnOp([scan.getResult(0)]);
  });
}

describe('nested control-flow forward + autodiff (if inside scan)', () => {
  const x = new Float32Array([1.0, -2.0, 3.0, -0.5]);
  const pred = new Float32Array([1, -1, 1, -1]);

  function reference(xv) {
    let acc = 0;
    for (let i = 0; i < T; i++) acc += pred[i] > 0 ? xv[i] * xv[i] : xv[i];
    return acc;
  }

  it('forward executes the nested if inside the scan body correctly', () => {
    const c = compileGraph(ifInScan('f'), CPUTarget());
    const out = new Float32Array(1);
    c.run('f', x, pred, out);
    expect(out[0]).toBeCloseTo(reference(x), 4);
  });

  it('backward gradient matches finite differences (gradcheck)', () => {
    const fwd = compileGraph(ifInScan('fwd'), CPUTarget());
    const runF = (xv) => { const o = new Float32Array(1); fwd.run('fwd', xv, pred, o); return o[0]; };

    const bwdRes = new BackwardGraphBuilder().build(ifInScan('bwd'));
    const bwdFunc = bwdRes.backwardFunc;
    const bc = compileGraph(bwdFunc, CPUTarget());
    const kname = bc.listKernels()[0];

    const inbufs = bwdFunc.inputTypes.map(ty => new Float32Array(Math.max(numel(ty.shape), 1)));
    inbufs[0][0] = 1;
    let seenT = 0;
    for (let i = 1; i < bwdFunc.inputTypes.length; i++) {
      if (numel(bwdFunc.inputTypes[i].shape) === T) inbufs[i].set(seenT++ === 0 ? x : pred);
    }

    const outbufs = bwdFunc.outputTypes.map(ty => new Float32Array(Math.max(numel(ty.shape), 1)));
    bc.run(kname, ...inbufs, ...outbufs);
    const gradX = outbufs[0];

    const eps = 1e-3;
    for (let i = 0; i < T; i++) {
      const xp = x.slice(); xp[i] += eps;
      const xm = x.slice(); xm[i] -= eps;
      const fd = (runF(xp) - runF(xm)) / (2 * eps);
      expect(gradX[i]).toBeCloseTo(fd, 2);
    }
  });
});
