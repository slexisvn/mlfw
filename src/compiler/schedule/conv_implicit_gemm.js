import {
  ForNode, ForKind, SeqNode, AllocateNode, LetStmtNode, IfThenElseNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode,
  MathOpNode, CompareNode, SyncThreadsNode,
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { findBlock, collectAllBlockNames } from '../autotune/block_analysis.js';
import { pickFixedConfig } from '../autotune/gpu_matmul_sketch.js';

const I = (v) => new IntImmNode(v);
const FZERO = () => new FloatImmNode(0);
const IV = (n) => new VariableNode(n, 'i32');
const ADD = (a, b) => new MathOpNode('+', a, b);
const SUB = (a, b) => new MathOpNode('-', a, b);
const MUL = (a, b) => new MathOpNode('*', a, b);
const DIV = (a, b) => new MathOpNode('//', a, b);
const MOD = (a, b) => new MathOpNode('%', a, b);
const LT = (a, b) => new CompareNode('lt', a, b);
const GE = (a, b) => new CompareNode('ge', a, b);
const AND = (a, b) => new MathOpNode('&&', a, b);
const forS = (v, e, b) => new ForNode(v, I(0), I(e), ForKind.SERIAL, b);
const forU = (v, e, b) => new ForNode(v, I(0), I(e), ForKind.UNROLLED, b);
const forT = (v, t, e, b) => new ForNode(v, I(0), I(e), ForKind.THREAD_BINDING, b, t);

export function detectPureConv(graphFunc) {
  let conv = null;
  for (const op of graphFunc.ops()) {
    if (op.opName === 'conv') { if (conv) return null; conv = op; }
    else if (op.opName !== 'return') return null;
  }
  if (!conv) return null;
  if ((conv.getAttr('groups') || 1) !== 1) return null;
  const il = conv.getAttr('input_layout');
  const kl = conv.getAttr('kernel_layout');
  if ((il && il !== 'NCHW') || (kl && kl !== 'OIHW')) return null;
  const inT = conv.getOperand(0).type, kerT = conv.getOperand(1).type, outT = conv.getResult(0).type;
  if (!inT || !kerT || !outT) return null;
  if (inT.shape.length !== 4 || kerT.shape.length !== 4 || outT.shape.length !== 4) return null;
  if (inT.dtype !== 'f32' || kerT.dtype !== 'f32' || outT.dtype !== 'f32') return null;
  const strides = conv.getAttr('strides') || [1, 1];
  const pad = conv.getAttr('padding') || [[0, 0], [0, 0]];
  const dil = conv.getAttr('dilation') || [1, 1];
  const N = inT.shape[0], Cin = inT.shape[1], H = inT.shape[2], W = inT.shape[3];
  const O = kerT.shape[0], Kh = kerT.shape[2], Kw = kerT.shape[3];
  const Oh = outT.shape[2], Ow = outT.shape[3];
  if (![N, Cin, H, W, O, Kh, Kw, Oh, Ow].every(d => typeof d === 'number' && d > 0)) return null;
  if (kerT.shape[1] !== Cin || outT.shape[1] !== O || outT.shape[0] !== N) return null;
  return {
    N, Cin, H, W, O, Kh, Kw, Oh, Ow,
    sH: strides[0], sW: strides[1], pH: pad[0][0], pW: pad[1][0], dH: dil[0], dW: dil[1],
  };
}

export function buildImplicitGemmConv(bufs, ci, params) {
  const { weight, input, output } = bufs;
  const { N, Cin, H, W, O, Kh, Kw, Oh, Ow, sH, sW, pH, pW, dH, dW } = ci;
  const { BM, BN, BK, TM, TN } = params;
  const M = O, GN = N * Oh * Ow, K = Cin * Kh * Kw;
  const KhKw = Kh * Kw, OhOw = Oh * Ow;
  const tX = BN / TN, tY = BM / TM, numThreads = tX * tY, numKTiles = Math.ceil(K / BK);
  const aTotal = BM * BK, bTotal = BK * BN;
  const aLoads = Math.ceil(aTotal / numThreads), bLoads = Math.ceil(bTotal / numThreads);

  const As = new Buffer('ig_As', [aTotal], 'f32', 'shared');
  const Bs = new Buffer('ig_Bs', [bTotal], 'f32', 'shared');
  const acc = new Buffer('ig_acc', [TM * TN], 'f32', 'local');
  const af = new Buffer('ig_af', [TM], 'f32', 'local');
  const bf = new Buffer('ig_bf', [TN], 'f32', 'local');

  const bx = IV('ig_bx'), by = IV('ig_by'), tx = IV('ig_tx'), ty = IV('ig_ty');
  const tid = IV('ig_tid'), brow = IV('ig_brow'), bcol = IV('ig_bcol'), k0 = IV('ig_k0');
  const accIdx = (mi, ni) => ADD(MUL(mi, I(TN)), ni);

  const im = IV('ig_im'), inn = IV('ig_in');
  const initAcc = forU(im, TM, forU(inn, TN, new BufferStoreNode(acc, [accIdx(im, inn)], FZERO())));

  const guardRow = M % BM !== 0, guardK = K % BK !== 0, guardCol = GN % BN !== 0;
  const guardALoad = aTotal % numThreads !== 0, guardBLoad = bTotal % numThreads !== 0;

  const la = IV('ig_la'), aidx = IV('ig_aidx');
  const aM = ADD(brow, DIV(aidx, I(BK)));
  const aK = ADD(k0, MOD(aidx, I(BK)));
  const aCin = DIV(aK, I(KhKw)), aR = MOD(aK, I(KhKw)), aKh = DIV(aR, I(Kw)), aKw = MOD(aR, I(Kw));
  let aVal = new BufferLoadNode(weight, [aM, aCin, aKh, aKw]);
  if (guardRow || guardK) {
    let c = guardRow ? LT(aM, I(M)) : null;
    if (guardK) c = c ? AND(c, LT(aK, I(K))) : LT(aK, I(K));
    aVal = new IfThenElseNode(c, aVal, FZERO());
  }
  let aStore = new BufferStoreNode(As, [aidx], aVal);
  if (guardALoad) aStore = new IfThenElseNode(LT(aidx, I(aTotal)), aStore);
  const loadA = forS(la, aLoads, new LetStmtNode(aidx, ADD(tid, MUL(la, I(numThreads))), aStore));

  const lb = IV('ig_lb'), bidx = IV('ig_bidx');
  const bK = ADD(k0, DIV(bidx, I(BN)));
  const bN = ADD(bcol, MOD(bidx, I(BN)));
  const bCin = DIV(bK, I(KhKw)), bR = MOD(bK, I(KhKw)), bKh = DIV(bR, I(Kw)), bKw = MOD(bR, I(Kw));
  const nB = DIV(bN, I(OhOw)), nR = MOD(bN, I(OhOw)), oh = DIV(nR, I(Ow)), ow = MOD(nR, I(Ow));
  const ih = SUB(ADD(MUL(oh, I(sH)), MUL(bKh, I(dH))), I(pH));
  const iw = SUB(ADD(MUL(ow, I(sW)), MUL(bKw, I(dW))), I(pW));
  let bVal = new BufferLoadNode(input, [nB, bCin, ih, iw]);
  let inb = AND(AND(GE(ih, I(0)), LT(ih, I(H))), AND(GE(iw, I(0)), LT(iw, I(W))));
  if (guardK) inb = AND(inb, LT(bK, I(K)));
  if (guardCol) inb = AND(inb, LT(bN, I(GN)));
  bVal = new IfThenElseNode(inb, bVal, FZERO());
  let bStore = new BufferStoreNode(Bs, [bidx], bVal);
  if (guardBLoad) bStore = new IfThenElseNode(LT(bidx, I(bTotal)), bStore);
  const loadB = forS(lb, bLoads, new LetStmtNode(bidx, ADD(tid, MUL(lb, I(numThreads))), bStore));

  const kk = IV('ig_kk'), fi = IV('ig_fi'), fj = IV('ig_fj');
  const fragA = forU(fi, TM, new BufferStoreNode(af, [fi],
    new BufferLoadNode(As, [ADD(MUL(ADD(MUL(ty, I(TM)), fi), I(BK)), kk)])));
  const fragB = forU(fj, TN, new BufferStoreNode(bf, [fj],
    new BufferLoadNode(Bs, [ADD(MUL(kk, I(BN)), ADD(MUL(tx, I(TN)), fj))])));
  const mi = IV('ig_mi'), ni = IV('ig_ni');
  const mma = forU(mi, TM, forU(ni, TN, new BufferStoreNode(acc, [accIdx(mi, ni)],
    ADD(new BufferLoadNode(acc, [accIdx(mi, ni)]),
        MUL(new BufferLoadNode(af, [mi]), new BufferLoadNode(bf, [ni]))))));
  const compute = forU(kk, BK, new SeqNode([fragA, fragB, mma]));

  const ktVar = IV('ig_kt');
  const ktBody = new LetStmtNode(k0, MUL(ktVar, I(BK)),
    new SeqNode([loadA, loadB, new SyncThreadsNode(), compute, new SyncThreadsNode()]));
  const ktLoop = forS(ktVar, numKTiles, ktBody);

  const wm = IV('ig_wm'), wn = IV('ig_wn');
  const cM = ADD(ADD(brow, MUL(ty, I(TM))), wm);
  const cN = ADD(ADD(bcol, MUL(tx, I(TN))), wn);
  const wB = DIV(cN, I(OhOw)), wR = MOD(cN, I(OhOw)), wOh = DIV(wR, I(Ow)), wOw = MOD(wR, I(Ow));
  let writeStore = new BufferStoreNode(output, [wB, cM, wOh, wOw], new BufferLoadNode(acc, [accIdx(wm, wn)]));
  if (guardRow || guardCol) {
    let c = guardRow ? LT(cM, I(M)) : null;
    if (guardCol) c = c ? AND(c, LT(cN, I(GN))) : LT(cN, I(GN));
    writeStore = new IfThenElseNode(c, writeStore);
  }
  const writeLoop = forU(wm, TM, forU(wn, TN, writeStore));

  const perThread = new SeqNode([initAcc, ktLoop, writeLoop]);
  const locals = new AllocateNode(acc, 'local', new AllocateNode(af, 'local', new AllocateNode(bf, 'local', perThread)));
  const letChain = new LetStmtNode(tid, ADD(MUL(ty, I(tX)), tx),
    new LetStmtNode(brow, MUL(by, I(BM)), new LetStmtNode(bcol, MUL(bx, I(BN)), locals)));
  const gridX = Math.ceil(GN / BN), gridY = Math.ceil(M / BM);
  const threadsNest = forT(by, 'blockIdx.y', gridY,
    forT(bx, 'blockIdx.x', gridX,
      forT(ty, 'threadIdx.y', tY,
        forT(tx, 'threadIdx.x', tX, letChain))));
  return new AllocateNode(As, 'shared', new AllocateNode(Bs, 'shared', threadsNest));
}

export function applyImplicitGemmConv(schedule, target) {
  const pf = schedule.func;
  const ci = pf.convInfo;
  if (!ci) return false;
  const names = collectAllBlockNames(pf.body);
  const accName = names.find(n => /^conv_acc_/.test(n));
  if (!accName) return false;
  const block = findBlock(pf.body, accName);
  if (!block || block.reads.length < 2 || block.writes.length < 1) return false;
  const input = block.reads[0].buffer, weight = block.reads[1].buffer, output = block.writes[0].buffer;
  if (!input || !weight || !output) return false;
  const M = ci.O, GN = ci.N * ci.Oh * ci.Ow, K = ci.Cin * ci.Kh * ci.Kw;
  if (K < 128 || GN < 64) return false;
  const cfg = pickFixedConfig(target, { M, N: GN, K });
  if (!cfg) return false;
  const body = buildImplicitGemmConv({ weight, input, output }, ci, cfg);
  schedule.func.body = body;
  if (schedule.func._setChild) schedule.func._setChild('body', body);
  schedule.func.gpuRegisterBlocked = true;
  return true;
}
