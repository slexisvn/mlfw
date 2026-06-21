import {
  BlockNode, SeqNode, AllocateNode, BufferStoreNode, BufferLoadNode,
  ForNode, VariableNode, IntImmNode, FloatImmNode, MathOpNode, CallExternNode, ForKind,
  IfThenElseNode, LetStmtNode, CompareNode
} from '../../../ir/tensor/nodes.js';
import { Buffer } from '../../../ir/tensor/buffer.js';
import { registerLoweringRule } from '../lowering_registry.js';

const Z = new IntImmNode(0);
const ld = (b, idx) => new BufferLoadNode(b, idx);
const st = (b, idx, val) => new BufferStoreNode(b, idx, val);
const mop = (o, a, b) => new MathOpNode(o, a, b);
const ext = (name, ...args) => new CallExternNode(name, args);
const forL = (v, n, body, kind = ForKind.SERIAL) => new ForNode(v, Z, new IntImmNode(n), kind, body);

export function register() {
  registerLoweringRule('scaled_dot_product_attention', (ctx, op, inputs, outputs) => {
    if (op.getAttr('causal')) throw new Error('causal flash attention lowering not yet supported');
    const [Q, K, V] = inputs;
    const O = outputs[0];
    const scale = op.getAttr('scale');
    const dtype = Q.dtype;
    const B = Q.shape[0], H = Q.shape[1], Lq = Q.shape[2], Dk = Q.shape[3];
    const Lk = K.shape[2], Dv = V.shape[3];

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

    const jLoop = forL(j, Lk, new SeqNode([sRed, scalarUpd, oUpd, mUpd]));
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
  });
}
