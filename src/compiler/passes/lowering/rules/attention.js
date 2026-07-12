import {
  BlockNode, SeqNode, AllocateNode, BufferStoreNode, BufferLoadNode,
  ForNode, VariableNode, IntImmNode, FloatImmNode, MathOpNode, CallExternNode, ForKind,
  IfThenElseNode, LetStmtNode, CompareNode, SyncThreadsNode
} from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import { registerLoweringRule, registerTargetLoweringRule } from '../lowering_registry.js';

const Z = new IntImmNode(0);
const I = (n) => new IntImmNode(n);
const ld = (b, idx) => new BufferLoadNode(b, idx);
const st = (b, idx, val) => new BufferStoreNode(b, idx, val);
const mop = (o, a, b) => new MathOpNode(o, a, b);
const ext = (name, ...args) => new CallExternNode(name, args);
const forL = (v, n, body, kind = ForKind.SERIAL) => new ForNode(v, Z, new IntImmNode(n), kind, body);
const forT = (v, tag, n, body) => new ForNode(v, Z, new IntImmNode(n), ForKind.THREAD_BINDING, body, tag);

function dims(op, inputs) {
  const [Q, K, V] = inputs;
  return {
    Q, K, V, O: null,
    scale: op.getAttr('scale'),
    causal: !!op.getAttr('causal'),
    dtype: Q.dtype,
    B: Q.shape[0], H: Q.shape[1], Lq: Q.shape[2], Dk: Q.shape[3],
    Lk: K.shape[2], Dv: V.shape[3],
  };
}

function buildNaive(ctx, op, inputs, outputs) {
  const { Q, K, V, scale, causal, dtype, B, H, Lq, Dk, Lk, Dv } = dims(op, inputs);
  const O = outputs[0];

  const m = new Buffer(ctx.blockName('fa_m'), [1], dtype, 'local');
  const l = new Buffer(ctx.blockName('fa_l'), [1], dtype, 'local');
  const s = new Buffer(ctx.blockName('fa_s'), [1], dtype, 'local');
  const mn = new Buffer(ctx.blockName('fa_mn'), [1], dtype, 'local');
  const p = new Buffer(ctx.blockName('fa_p'), [1], dtype, 'local');
  const cr = new Buffer(ctx.blockName('fa_cr'), [1], dtype, 'local');
  const o = new Buffer(ctx.blockName('fa_o'), [Dv], dtype, 'local');

  const b = new VariableNode(ctx.blockName('fa_b'), 'int32');
  const h = new VariableNode(ctx.blockName('fa_h'), 'int32');
  const i = new VariableNode(ctx.blockName('fa_i'), 'i32');
  const j = new VariableNode(ctx.blockName('fa_j'), 'int32');
  const d = new VariableNode(ctx.blockName('fa_d'), 'int32');
  const dInit = new VariableNode(ctx.blockName('fa_di'), 'int32');
  const dO = new VariableNode(ctx.blockName('fa_do'), 'int32');
  const dW = new VariableNode(ctx.blockName('fa_dw'), 'int32');

  const initM = new BlockNode(ctx.blockName('fa_initm'), [], [], [{ buffer: m }], st(m, [Z], new FloatImmNode(-Infinity)));
  const initL = new BlockNode(ctx.blockName('fa_initl'), [], [], [{ buffer: l }], st(l, [Z], new FloatImmNode(0)));
  const initO = forL(dInit, Dv, new BlockNode(ctx.blockName('fa_inito'), [], [], [{ buffer: o }], st(o, [dInit], new FloatImmNode(0))));

  const sInit = new BlockNode(ctx.blockName('fa_sinit'), [], [], [{ buffer: s }], st(s, [Z], new FloatImmNode(0)));
  const sAccum = forL(d, Dk, new BlockNode(ctx.blockName('fa_sred'), [],
    [{ buffer: Q }, { buffer: K }, { buffer: s }], [{ buffer: s }],
    st(s, [Z], mop('+', ld(s, [Z]), mop('*', ld(Q, [b, h, i, d]), ld(K, [b, h, j, d]))))));
  const sRed = new SeqNode([sInit, sAccum]);

  const scalarUpd = new BlockNode(ctx.blockName('fa_supd'), [],
    [{ buffer: s }, { buffer: m }, { buffer: l }],
    [{ buffer: s }, { buffer: mn }, { buffer: p }, { buffer: cr }, { buffer: l }],
    new SeqNode([
      st(s, [Z], mop('*', ld(s, [Z]), new FloatImmNode(scale))),
      st(mn, [Z], ext('max', ld(m, [Z]), ld(s, [Z]))),
      st(p, [Z], ext('exp', mop('-', ld(s, [Z]), ld(mn, [Z])))),
      st(cr, [Z], ext('exp', mop('-', ld(m, [Z]), ld(mn, [Z])))),
      st(l, [Z], mop('+', mop('*', ld(l, [Z]), ld(cr, [Z])), ld(p, [Z]))),
    ]));

  const oUpd = forL(dO, Dv, new BlockNode(ctx.blockName('fa_oupd'), [],
    [{ buffer: o }, { buffer: cr }, { buffer: p }, { buffer: V }], [{ buffer: o }],
    st(o, [dO], mop('+', mop('*', ld(o, [dO]), ld(cr, [Z])), mop('*', ld(p, [Z]), ld(V, [b, h, j, dO]))))));

  const mUpd = new BlockNode(ctx.blockName('fa_mupd'), [], [{ buffer: mn }], [{ buffer: m }], st(m, [Z], ld(mn, [Z])));

  const jStep = new SeqNode([sRed, scalarUpd, oUpd, mUpd]);
  const causalOffset = Lk - Lq;
  const jGuarded = causal
    ? new IfThenElseNode(new CompareNode('le', j, mop('+', i, I(causalOffset))), jStep)
    : jStep;
  const jLoop = forL(j, Lk, jGuarded);
  const finalW = forL(dW, Dv, new BlockNode(ctx.blockName('fa_finw'), [],
    [{ buffer: o }, { buffer: l }], [{ buffer: O }], st(O, [b, h, i, dW], mop('/', ld(o, [dW]), ld(l, [Z])))));

  let perQuery = new SeqNode([initM, initL, initO, jLoop, finalW]);
  for (const buf of [o, cr, p, mn, s, l, m]) perQuery = new AllocateNode(buf, buf.scope, perQuery);

  const TPB = Math.min(Lq, 256);
  const nBlk = Math.ceil(Lq / TPB);
  let queryLoop;
  if (nBlk === 1) {
    queryLoop = new ForNode(i, Z, new IntImmNode(Lq), ForKind.THREAD_BINDING, perQuery, 'threadIdx.x');
  } else {
    const iInner = new VariableNode(ctx.blockName('fa_ii'), 'i32');
    const iOuter = new VariableNode(ctx.blockName('fa_io'), 'i32');
    const iExpr = mop('+', mop('*', iOuter, new IntImmNode(TPB)), iInner);
    const guarded = nBlk * TPB === Lq ? perQuery
      : new IfThenElseNode(new CompareNode('lt', i, new IntImmNode(Lq)), perQuery);
    const inner = new ForNode(iInner, Z, new IntImmNode(TPB), ForKind.THREAD_BINDING, new LetStmtNode(i, iExpr, guarded), 'threadIdx.x');
    queryLoop = new ForNode(iOuter, Z, new IntImmNode(nBlk), ForKind.THREAD_BINDING, inner, 'blockIdx.z');
  }
  const hLoop = new ForNode(h, Z, new IntImmNode(H), ForKind.THREAD_BINDING, queryLoop, 'blockIdx.y');
  return new ForNode(b, Z, new IntImmNode(B), ForKind.THREAD_BINDING, hLoop, 'blockIdx.x');
}

function buildTiled(ctx, op, inputs, outputs, TPB) {
  const { Q, K, V, scale, causal, dtype, B, H, Lq, Dk, Lk, Dv } = dims(op, inputs);
  const O = outputs[0];
  const nKBlk = Math.ceil(Lk / TPB);
  const nQBlk = Math.ceil(Lq / TPB);
  const causalOffset = Lk - Lq;

  const Ks = new Buffer(ctx.blockName('fa_Ks'), [TPB, Dk], dtype, 'shared');
  const Vs = new Buffer(ctx.blockName('fa_Vs'), [TPB, Dv], dtype, 'shared');
  const qreg = new Buffer(ctx.blockName('fa_q'), [Dk], dtype, 'local');
  const m = new Buffer(ctx.blockName('fa_m'), [1], dtype, 'local');
  const l = new Buffer(ctx.blockName('fa_l'), [1], dtype, 'local');
  const s = new Buffer(ctx.blockName('fa_s'), [1], dtype, 'local');
  const mn = new Buffer(ctx.blockName('fa_mn'), [1], dtype, 'local');
  const p = new Buffer(ctx.blockName('fa_p'), [1], dtype, 'local');
  const cr = new Buffer(ctx.blockName('fa_cr'), [1], dtype, 'local');
  const o = new Buffer(ctx.blockName('fa_o'), [Dv], dtype, 'local');

  const b = new VariableNode(ctx.blockName('fa_b'), 'int32');
  const h = new VariableNode(ctx.blockName('fa_h'), 'int32');
  const qb = new VariableNode(ctx.blockName('fa_qb'), 'int32');
  const t = new VariableNode(ctx.blockName('fa_t'), 'int32');
  const i = new VariableNode(ctx.blockName('fa_i'), 'i32');
  const kt = new VariableNode(ctx.blockName('fa_kt'), 'int32');
  const jj = new VariableNode(ctx.blockName('fa_jj'), 'int32');
  const dq = new VariableNode(ctx.blockName('fa_dq'), 'int32');
  const dk = new VariableNode(ctx.blockName('fa_dk'), 'int32');
  const dv = new VariableNode(ctx.blockName('fa_dv'), 'int32');
  const ds = new VariableNode(ctx.blockName('fa_ds'), 'int32');
  const doo = new VariableNode(ctx.blockName('fa_doo'), 'int32');
  const dw = new VariableNode(ctx.blockName('fa_dw'), 'int32');
  const dinit = new VariableNode(ctx.blockName('fa_din'), 'int32');

  const iExpr = mop('+', mop('*', qb, I(TPB)), t);
  const kj = mop('+', mop('*', kt, I(TPB)), t);
  const jGlobal = mop('+', mop('*', kt, I(TPB)), jj);
  const inQ = new CompareNode('lt', i, I(Lq));
  const inK = new CompareNode('lt', kj, I(Lk));

  const loadQ = forL(dq, Dk, st(qreg, [dq], ld(Q, [b, h, i, dq])));
  const initM = st(m, [Z], new FloatImmNode(-Infinity));
  const initL = st(l, [Z], new FloatImmNode(0));
  const initO = forL(dinit, Dv, st(o, [dinit], new FloatImmNode(0)));
  const prologue = new IfThenElseNode(inQ, new SeqNode([loadQ, initM, initL, initO]));

  const loadK = new IfThenElseNode(inK, forL(dk, Dk, st(Ks, [t, dk], ld(K, [b, h, kj, dk]))));
  const loadV = new IfThenElseNode(inK, forL(dv, Dv, st(Vs, [t, dv], ld(V, [b, h, kj, dv]))));

  const sInit = st(s, [Z], new FloatImmNode(0));
  const sAccum = forL(ds, Dk, st(s, [Z], mop('+', ld(s, [Z]), mop('*', ld(qreg, [ds]), ld(Ks, [jj, ds])))));
  const scalarUpd = new SeqNode([
    st(s, [Z], mop('*', ld(s, [Z]), new FloatImmNode(scale))),
    st(mn, [Z], ext('max', ld(m, [Z]), ld(s, [Z]))),
    st(p, [Z], ext('exp', mop('-', ld(s, [Z]), ld(mn, [Z])))),
    st(cr, [Z], ext('exp', mop('-', ld(m, [Z]), ld(mn, [Z])))),
    st(l, [Z], mop('+', mop('*', ld(l, [Z]), ld(cr, [Z])), ld(p, [Z]))),
  ]);
  const oUpd = forL(doo, Dv, st(o, [doo], mop('+', mop('*', ld(o, [doo]), ld(cr, [Z])), mop('*', ld(p, [Z]), ld(Vs, [jj, doo])))));
  const mUpd = st(m, [Z], ld(mn, [Z]));
  const jStep = new SeqNode([sInit, sAccum, scalarUpd, oUpd, mUpd]);

  const inJ = new CompareNode('lt', jGlobal, I(Lk));
  const jInner = causal
    ? new IfThenElseNode(new CompareNode('le', jGlobal, mop('+', i, I(causalOffset))), jStep)
    : jStep;
  const jjLoop = forL(jj, TPB, new IfThenElseNode(inJ, jInner));
  const compute = new IfThenElseNode(inQ, jjLoop);

  const ktBody = new SeqNode([loadK, loadV, new SyncThreadsNode(), compute, new SyncThreadsNode()]);
  const ktLoop = forL(kt, nKBlk, ktBody);

  const writeBack = new IfThenElseNode(inQ, forL(dw, Dv, st(O, [b, h, i, dw], mop('/', ld(o, [dw]), ld(l, [Z])))));

  let perThread = new LetStmtNode(i, iExpr, new SeqNode([prologue, ktLoop, writeBack]));
  for (const buf of [o, cr, p, mn, s, l, m, qreg]) perThread = new AllocateNode(buf, buf.scope, perThread);

  const tLoop = forT(t, 'threadIdx.x', TPB, perThread);
  const shared = new AllocateNode(Ks, 'shared', new AllocateNode(Vs, 'shared', tLoop));
  const qbLoop = forT(qb, 'blockIdx.z', nQBlk, shared);
  const hLoop = forT(h, 'blockIdx.y', H, qbLoop);
  return forT(b, 'blockIdx.x', B, hLoop);
}

export function register() {
  registerLoweringRule('scaled_dot_product_attention', buildNaive);

  registerTargetLoweringRule('scaled_dot_product_attention', 'cuda', (ctx, op, inputs, outputs) => {
    const { Lq, Dk, Dv, causal } = dims(op, inputs);
    const budget = 32 * 1024;
    const TPB = Math.min(128, Lq, Math.floor(budget / ((Dk + Dv) * 4 * 2)));
    if (!causal || TPB < 8) return buildNaive(ctx, op, inputs, outputs);
    return buildTiled(ctx, op, inputs, outputs, TPB);
  });
}
