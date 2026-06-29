import {
  ForNode, ForKind, SeqNode, AllocateNode, LetStmtNode, IfThenElseNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode,
  MathOpNode, CompareNode, SyncThreadsNode, VecCopyNode,
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

const VEC = 4;

export function vectorizableConvConfig(target, ci, cfg) {
  if (!validConvConfig(target, { M: ci.O, N: ci.N * ci.Oh * ci.Ow, K: ci.Cin * ci.Kh * ci.Kw }, cfg)) return false;
  const { BM, BN, BK, TM, TN } = cfg;
  const M = ci.O, GN = ci.N * ci.Oh * ci.Ow, K = ci.Cin * ci.Kh * ci.Kw;
  if (M % BM !== 0 || GN % BN !== 0 || K % BK !== 0) return false;
  if ([BM, BN, BK, TM, TN].some(v => v % VEC !== 0)) return false;
  const numThreads = (BM / TM) * (BN / TN);
  if ((BM * BK) % (VEC * numThreads) !== 0) return false;
  if ((BK * BN) % numThreads !== 0) return false;
  return true;
}

export function buildVectorizedImplicitGemmConv(bufs, ci, params) {
  const { weight, input, output } = bufs;
  const { N, Cin, H, W, O, Kh, Kw, Oh, Ow, sH, sW, pH, pW, dH, dW } = ci;
  const { BM, BN, BK, TM, TN } = params;
  const V = VEC;
  const M = O, GN = N * Oh * Ow, K = Cin * Kh * Kw;
  const KhKw = Kh * Kw, OhOw = Oh * Ow, HW = H * W, CHW = Cin * H * W;
  const tX = BN / TN, tY = BM / TM, numThreads = tX * tY, numKTiles = K / BK;
  const aTotal = BM * BK, bTotal = BK * BN;
  const aRegs = aTotal / numThreads, bLoads = bTotal / numThreads;
  const aLoadsV = aRegs / V;
  const BKv = BK / V;

  const maxIh = (Oh - 1) * sH + (Kh - 1) * dH - pH, maxIw = (Ow - 1) * sW + (Kw - 1) * dW - pW;
  const needGuardH = pH > 0 || maxIh >= H, needGuardW = pW > 0 || maxIw >= W;

  const weightFlat = new Buffer(weight.name, [M * K > 0 ? M * K : 1], weight.dtype, weight.scope);
  const inputFlat = new Buffer(input.name, [N * CHW > 0 ? N * CHW : 1], input.dtype, input.scope);

  const As = new Buffer('iv_As', [2 * aTotal], 'f32', 'shared'); As.align16 = true;
  const Bs = new Buffer('iv_Bs', [2 * bTotal], 'f32', 'shared'); Bs.align16 = true;
  const acc = new Buffer('iv_acc', [TM * TN], 'f32', 'local');
  const af = new Buffer('iv_af', [TM], 'f32', 'local'); af.align16 = true;
  const bf = new Buffer('iv_bf', [TN], 'f32', 'local'); bf.align16 = true;
  const ra = new Buffer('iv_ra', [aRegs], 'f32', 'local'); ra.align16 = true;
  const rb = new Buffer('iv_rb', [bLoads], 'f32', 'local');

  const bx = IV('iv_bx'), by = IV('iv_by'), tx = IV('iv_tx'), ty = IV('iv_ty');
  const tid = IV('iv_tid'), brow = IV('iv_brow'), bcol = IV('iv_bcol');
  const accIdx = (mi, ni) => ADD(MUL(mi, I(TN)), ni);
  let uid = 0;

  const im = IV('iv_im'), inn = IV('iv_in');
  const initAcc = forU(im, TM, forU(inn, TN, new BufferStoreNode(acc, [accIdx(im, inn)], FZERO())));

  const prefetch = (k0e) => {
    const u = uid++;
    const la = IV('iv_la' + u), c = IV('iv_c' + u);
    const m = DIV(c, I(BKv)), kc = MUL(MOD(c, I(BKv)), I(V));
    const pa = forU(la, aLoadsV, new LetStmtNode(c, ADD(tid, MUL(la, I(numThreads))),
      new VecCopyNode(ra, MUL(la, I(V)), weightFlat, ADD(MUL(ADD(brow, m), I(K)), ADD(k0e, kc)), V)));
    const lb = IV('iv_lb' + u), e = IV('iv_e' + u);
    const kRow = DIV(e, I(BN)), col = MOD(e, I(BN)), bN = ADD(bcol, col);
    const bK = ADD(k0e, kRow);
    const bCin = DIV(bK, I(KhKw)), bR = MOD(bK, I(KhKw)), bKh = DIV(bR, I(Kw)), bKw = MOD(bR, I(Kw));
    const nB = DIV(bN, I(OhOw)), nR = MOD(bN, I(OhOw)), oh = DIV(nR, I(Ow)), ow = MOD(nR, I(Ow));
    const ih = SUB(ADD(MUL(oh, I(sH)), MUL(bKh, I(dH))), I(pH));
    const iw = SUB(ADD(MUL(ow, I(sW)), MUL(bKw, I(dW))), I(pW));
    const off = ADD(ADD(ADD(MUL(nB, I(CHW)), MUL(bCin, I(HW))), MUL(ih, I(W))), iw);
    let v = new BufferLoadNode(inputFlat, [off]);
    let inb = null;
    if (needGuardH) inb = AND(GE(ih, I(0)), LT(ih, I(H)));
    if (needGuardW) { const cc = AND(GE(iw, I(0)), LT(iw, I(W))); inb = inb ? AND(inb, cc) : cc; }
    if (inb) v = new IfThenElseNode(inb, v, FZERO());
    const pb = forU(lb, bLoads, new LetStmtNode(e, ADD(tid, MUL(lb, I(numThreads))), new BufferStoreNode(rb, [lb], v)));
    return new SeqNode([pa, pb]);
  };

  const commit = (offAe, offBe) => {
    const u = uid++;
    const la = IV('iv_la' + u), c = IV('iv_c' + u);
    const m = DIV(c, I(BKv)), kc = MUL(MOD(c, I(BKv)), I(V));
    const stores = [];
    for (let i = 0; i < V; i++) {
      stores.push(new BufferStoreNode(As, [ADD(offAe, ADD(MUL(ADD(kc, I(i)), I(BM)), m))], new BufferLoadNode(ra, [ADD(MUL(la, I(V)), I(i))])));
    }
    const ca = forU(la, aLoadsV, new LetStmtNode(c, ADD(tid, MUL(la, I(numThreads))), new SeqNode(stores)));
    const lb = IV('iv_lb' + u), e = IV('iv_e' + u);
    const cb = forU(lb, bLoads, new LetStmtNode(e, ADD(tid, MUL(lb, I(numThreads))),
      new BufferStoreNode(Bs, [ADD(offBe, e)], new BufferLoadNode(rb, [lb]))));
    return new SeqNode([ca, cb]);
  };

  const computeMMA = (offAe, offBe) => {
    const u = uid++;
    const kk = IV('iv_kk' + u);
    const frags = [];
    const aBase = ADD(offAe, ADD(MUL(kk, I(BM)), MUL(ty, I(TM))));
    for (let q = 0; q < TM / V; q++) frags.push(new VecCopyNode(af, I(q * V), As, ADD(aBase, I(q * V)), V));
    const bBaseE = ADD(offBe, ADD(MUL(kk, I(BN)), MUL(tx, I(TN))));
    for (let q = 0; q < TN / V; q++) frags.push(new VecCopyNode(bf, I(q * V), Bs, ADD(bBaseE, I(q * V)), V));
    const mi = IV('iv_mi' + u), ni = IV('iv_ni' + u);
    const mma = forU(mi, TM, forU(ni, TN, new BufferStoreNode(acc, [accIdx(mi, ni)],
      ADD(new BufferLoadNode(acc, [accIdx(mi, ni)]),
          MUL(new BufferLoadNode(af, [mi]), new BufferLoadNode(bf, [ni]))))));
    return forU(kk, BK, new SeqNode([...frags, mma]));
  };

  const ktVar = IV('iv_kt');
  const p = IV('iv_p'), pN = IV('iv_pN');
  const offA = MUL(p, I(aTotal)), offB = MUL(p, I(bTotal));
  const offAn = MUL(pN, I(aTotal)), offBn = MUL(pN, I(bTotal));
  const isNotLast = LT(ktVar, I(numKTiles - 1));
  const preamble = new SeqNode([prefetch(I(0)), commit(I(0), I(0)), new SyncThreadsNode()]);
  const ktBody = new LetStmtNode(p, MOD(ktVar, I(2)), new LetStmtNode(pN, MOD(ADD(ktVar, I(1)), I(2)),
    new SeqNode([
      new IfThenElseNode(isNotLast, prefetch(MUL(ADD(ktVar, I(1)), I(BK)))),
      computeMMA(offA, offB),
      new IfThenElseNode(isNotLast, new SeqNode([commit(offAn, offBn), new SyncThreadsNode()])),
    ])));
  const ktLoop = forS(ktVar, numKTiles, ktBody);

  const wm = IV('iv_wm'), wn = IV('iv_wn');
  const cM = ADD(ADD(brow, MUL(ty, I(TM))), wm);
  const cN = ADD(ADD(bcol, MUL(tx, I(TN))), wn);
  const wB = DIV(cN, I(OhOw)), wR = MOD(cN, I(OhOw)), wOh = DIV(wR, I(Ow)), wOw = MOD(wR, I(Ow));
  const writeStore = new BufferStoreNode(output, [wB, cM, wOh, wOw], new BufferLoadNode(acc, [accIdx(wm, wn)]));
  const writeLoop = forU(wm, TM, forU(wn, TN, writeStore));

  const perThread = new SeqNode([initAcc, preamble, ktLoop, writeLoop]);
  const locals = new AllocateNode(acc, 'local', new AllocateNode(af, 'local', new AllocateNode(bf, 'local',
    new AllocateNode(ra, 'local', new AllocateNode(rb, 'local', perThread)))));
  const letChain = new LetStmtNode(tid, ADD(MUL(ty, I(tX)), tx),
    new LetStmtNode(brow, MUL(by, I(BM)), new LetStmtNode(bcol, MUL(bx, I(BN)), locals)));
  const gridX = GN / BN, gridY = M / BM;
  const threadsNest = forT(by, 'blockIdx.y', gridY,
    forT(bx, 'blockIdx.x', gridX,
      forT(ty, 'threadIdx.y', tY,
        forT(tx, 'threadIdx.x', tX, letChain))));
  return new AllocateNode(As, 'shared', new AllocateNode(Bs, 'shared', threadsNest));
}

function validConvConfig(target, dims, cfg) {
  const { BM, BN, BK, TM, TN } = cfg;
  if (![BM, BN, BK, TM, TN].every(v => typeof v === 'number' && v > 0)) return false;
  if (BM % TM !== 0 || BN % TN !== 0) return false;
  const tX = BN / TN, tY = BM / TM, threads = tX * tY;
  const warp = target.warpSize || 32;
  if (threads % warp !== 0) return false;
  if (threads > (target.maxThreadsPerBlock || 1024)) return false;
  const smem = (BM * BK + BK * BN) * 4 * 2;
  if (smem > (target.sharedMemoryBytes || 49152)) return false;
  const regs = TM * TN + TM + TN + warp;
  if (regs > (target.registersPerThread || 255)) return false;
  return true;
}

const VEC_CONFIGS = [
  { BM: 128, BN: 64, BK: 8, TM: 8, TN: 8 },
  { BM: 64, BN: 64, BK: 8, TM: 8, TN: 8 },
  { BM: 64, BN: 64, BK: 8, TM: 4, TN: 8 },
  { BM: 64, BN: 32, BK: 8, TM: 8, TN: 8 },
  { BM: 32, BN: 64, BK: 8, TM: 4, TN: 8 },
];

function pickVectorizedConvConfig(target, ci) {
  for (const cfg of VEC_CONFIGS) {
    if (vectorizableConvConfig(target, ci, cfg)) return cfg;
  }
  return null;
}

export function applyImplicitGemmConv(schedule, target, sCfg) {
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
  const override = sCfg && sCfg.convConfig;
  const noVec = sCfg && sCfg.convNoVec;

  let body = null;
  if (!noVec) {
    const vecCfg = override
      ? (vectorizableConvConfig(target, ci, override) ? override : null)
      : pickVectorizedConvConfig(target, ci);
    if (vecCfg) body = buildVectorizedImplicitGemmConv({ weight, input, output }, ci, vecCfg);
  }
  if (!body) {
    const cfg = override
      ? (validConvConfig(target, { M, N: GN, K }, override) ? override : null)
      : pickFixedConfig(target, { M, N: GN, K });
    if (!cfg) return false;
    body = buildImplicitGemmConv({ weight, input, output }, ci, cfg);
  }
  schedule.func.body = body;
  if (schedule.func._setChild) schedule.func._setChild('body', body);
  schedule.func.gpuRegisterBlocked = true;
  return true;
}
