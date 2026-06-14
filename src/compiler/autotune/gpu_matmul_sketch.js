import {
  ForNode, ForKind, SeqNode, AllocateNode, LetStmtNode, IfThenElseNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode,
  MathOpNode, CompareNode, SyncThreadsNode, mathOp,
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { ScheduleSketch, SearchVariable } from './search_space.js';
import { classifyBlock } from '../schedule/rules.js';

const I = (v) => new IntImmNode(v);
const FZERO = () => new FloatImmNode(0);
const IV = (name) => new VariableNode(name, 'i32');
const ADD = (a, b) => mathOp('+', a, b);
const MUL = (a, b) => mathOp('*', a, b);
const DIV = (a, b) => mathOp('//', a, b);
const MOD = (a, b) => mathOp('%', a, b);
const LT = (a, b) => new CompareNode('lt', a, b);
const AND = (a, b) => new MathOpNode('&&', a, b);

const forS = (v, ext, body) => new ForNode(v, I(0), I(ext), ForKind.SERIAL, body);
const forU = (v, ext, body) => new ForNode(v, I(0), I(ext), ForKind.UNROLLED, body);
const forT = (v, tag, ext, body) => new ForNode(v, I(0), I(ext), ForKind.THREAD_BINDING, body, tag);

function pow2Range(min, max) {
  const out = [];
  for (let v = 1; v <= max; v *= 2) {
    if (v >= min) out.push(v);
  }
  return out;
}

export function matmulTileDims(primFunc, blockName) {
  const info = classifyBlock(primFunc, blockName);
  if (!info) return null;
  const block = findBlock(primFunc.body, blockName);
  if (!block || block.reads.length < 2 || block.writes.length < 1) return null;
  const A = block.reads[0].buffer;
  const B = block.reads[1].buffer;
  const C = block.writes[0].buffer;
  if (!A || !B || !C) return null;
  if (A.shape.length !== 2 || B.shape.length !== 2 || C.shape.length !== 2) return null;
  const M = C.shape[0], N = C.shape[1], K = A.shape[1];
  if (![M, N, K].every(d => typeof d === 'number' && d > 0)) return null;
  if (B.shape[0] !== K || B.shape[1] !== N) return null;
  if (A.dtype !== 'f32' || B.dtype !== 'f32' || C.dtype !== 'f32') return null;
  return { A, B, C, M, N, K };
}

function findBlock(root, name) {
  const stack = [root];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    if (n.type === 'BlockNode' && n.name === name) return n;
    if (n.body) stack.push(n.body);
    if (n.stmts) for (const s of n.stmts) stack.push(s);
    if (n.thenBody) stack.push(n.thenBody);
    if (n.elseBody) stack.push(n.elseBody);
    if (n.initBody) stack.push(n.initBody);
  }
  return null;
}

export function enumerateRegisterBlockConfigs(target, dims, maxCandidates = 32) {
  const maxThreads = target.maxThreadsPerBlock || 1024;
  const warp = target.warpSize || 32;
  const smemBytes = target.sharedMemoryBytes || 49152;
  const regs = target.registersPerThread || 255;
  const bytesPerEl = 4;
  const { M, N, K } = dims;

  const regTiles = pow2Range(2, 8);
  const threadAxis = pow2Range(2, warp / 2);
  const kTiles = pow2Range(warp / 4, warp / 2);
  const minThreads = 2 * warp;
  const maxBlockThreads = Math.min(maxThreads, 8 * warp);
  const capDim = 4 * warp;

  const seen = new Set();
  const configs = [];
  for (const TM of regTiles) {
    for (const TN of regTiles) {
      if (TM * TN > capDim) continue;
      if (TM * TN + TM + TN + warp > regs) continue;
      for (const tY of threadAxis) {
        for (const tX of threadAxis) {
          const threads = tX * tY;
          if (threads < minThreads || threads > maxBlockThreads) continue;
          if (threads % warp !== 0) continue;
          const BM = TM * tY;
          const BN = TN * tX;
          if (BM > M || BN > N || BM > capDim || BN > capDim) continue;
          for (const BK of kTiles) {
            if (BK > K) continue;
            const smem = (BM * BK + BK * BN) * bytesPerEl;
            if (smem > smemBytes) continue;
            const key = `${BM}_${BN}_${BK}_${TM}_${TN}`;
            if (seen.has(key)) continue;
            seen.add(key);
            configs.push({ BM, BN, BK, TM, TN, threads });
          }
        }
      }
    }
  }
  configs.sort((a, b) => goodness(b, warp) - goodness(a, warp));
  return configs.slice(0, maxCandidates);
}

function goodness(c, warp) {
  const reuse = c.TM * c.TN;
  const squareTile = -Math.abs(c.TM - c.TN);
  const squareBlock = -Math.abs(Math.log2(c.BM) - Math.log2(c.BN));
  const occ = c.threads >= 4 * warp && c.threads <= 8 * warp ? 1 : 0;
  const shallowK = c.BK === warp / 4 ? 1 : 0;
  return occ * 100 + reuse * 4 + squareTile * 6 + squareBlock * 4 + shallowK;
}

export function buildRegisterBlockedMatmul(bufs, params) {
  const { A, B, C, M, N, K } = bufs;
  const { BM, BN, BK, TM, TN } = params;
  const tX = BN / TN;
  const tY = BM / TM;
  const numThreads = tX * tY;
  const numKTiles = Math.ceil(K / BK);
  const aTotal = BM * BK;
  const bTotal = BK * BN;
  const aLoads = Math.ceil(aTotal / numThreads);
  const bLoads = Math.ceil(bTotal / numThreads);

  const guardRow = M % BM !== 0;
  const guardCol = N % BN !== 0;
  const guardK = K % BK !== 0;
  const guardALoad = aTotal % numThreads !== 0;
  const guardBLoad = bTotal % numThreads !== 0;

  const As = new Buffer('rb_As', [aTotal], 'f32', 'shared');
  const Bs = new Buffer('rb_Bs', [bTotal], 'f32', 'shared');
  const acc = new Buffer('rb_acc', [TM * TN], 'f32', 'local');
  const af = new Buffer('rb_af', [TM], 'f32', 'local');
  const bf = new Buffer('rb_bf', [TN], 'f32', 'local');

  const bx = IV('rb_bx'), by = IV('rb_by'), tx = IV('rb_tx'), ty = IV('rb_ty');
  const tid = IV('rb_tid'), brow = IV('rb_brow'), bcol = IV('rb_bcol');
  const k0 = IV('rb_k0');

  const accIdx = (mi, ni) => ADD(MUL(mi, I(TN)), ni);

  const im = IV('rb_im'), inn = IV('rb_in');
  const initAcc = forU(im, TM, forU(inn, TN,
    new BufferStoreNode(acc, [accIdx(im, inn)], FZERO())));

  const la = IV('rb_la'), aidx = IV('rb_aidx');
  const aRow = ADD(brow, DIV(aidx, I(BK)));
  const aCol = ADD(k0, MOD(aidx, I(BK)));
  let aVal = new BufferLoadNode(A, [aRow, aCol]);
  if (guardRow || guardK) {
    let cond = guardRow ? LT(aRow, I(M)) : null;
    if (guardK) cond = cond ? AND(cond, LT(aCol, I(K))) : LT(aCol, I(K));
    aVal = new IfThenElseNode(cond, aVal, FZERO());
  }
  let aStore = new BufferStoreNode(As, [aidx], aVal);
  if (guardALoad) aStore = new IfThenElseNode(LT(aidx, I(aTotal)), aStore);
  const loadA = forS(la, aLoads,
    new LetStmtNode(aidx, ADD(tid, MUL(la, I(numThreads))), aStore));

  const lb = IV('rb_lb'), bidx = IV('rb_bidx');
  const bRow = ADD(k0, DIV(bidx, I(BN)));
  const bCol = ADD(bcol, MOD(bidx, I(BN)));
  let bVal = new BufferLoadNode(B, [bRow, bCol]);
  if (guardK || guardCol) {
    let cond = guardK ? LT(bRow, I(K)) : null;
    if (guardCol) cond = cond ? AND(cond, LT(bCol, I(N))) : LT(bCol, I(N));
    bVal = new IfThenElseNode(cond, bVal, FZERO());
  }
  let bStore = new BufferStoreNode(Bs, [bidx], bVal);
  if (guardBLoad) bStore = new IfThenElseNode(LT(bidx, I(bTotal)), bStore);
  const loadB = forS(lb, bLoads,
    new LetStmtNode(bidx, ADD(tid, MUL(lb, I(numThreads))), bStore));

  const kk = IV('rb_kk'), fi = IV('rb_fi'), fj = IV('rb_fj');
  const fragA = forU(fi, TM, new BufferStoreNode(af, [fi],
    new BufferLoadNode(As, [ADD(MUL(ADD(MUL(ty, I(TM)), fi), I(BK)), kk)])));
  const fragB = forU(fj, TN, new BufferStoreNode(bf, [fj],
    new BufferLoadNode(Bs, [ADD(MUL(kk, I(BN)), ADD(MUL(tx, I(TN)), fj))])));
  const mi = IV('rb_mi'), ni = IV('rb_ni');
  const mma = forU(mi, TM, forU(ni, TN, new BufferStoreNode(acc, [accIdx(mi, ni)],
    ADD(new BufferLoadNode(acc, [accIdx(mi, ni)]),
        MUL(new BufferLoadNode(af, [mi]), new BufferLoadNode(bf, [ni]))))));
  const compute = forU(kk, BK, new SeqNode([fragA, fragB, mma]));

  const ktVar = IV('rb_kt');
  const ktBody = new LetStmtNode(k0, MUL(ktVar, I(BK)),
    new SeqNode([loadA, loadB, new SyncThreadsNode(), compute, new SyncThreadsNode()]));
  const ktLoop = forS(ktVar, numKTiles, ktBody);

  const wm = IV('rb_wm'), wn = IV('rb_wn');
  const rowC = ADD(ADD(brow, MUL(ty, I(TM))), wm);
  const colC = ADD(ADD(bcol, MUL(tx, I(TN))), wn);
  let writeStore = new BufferStoreNode(C, [rowC, colC], new BufferLoadNode(acc, [accIdx(wm, wn)]));
  if (guardRow || guardCol) {
    let cond = guardRow ? LT(rowC, I(M)) : null;
    if (guardCol) cond = cond ? AND(cond, LT(colC, I(N))) : LT(colC, I(N));
    writeStore = new IfThenElseNode(cond, writeStore);
  }
  const writeLoop = forU(wm, TM, forU(wn, TN, writeStore));

  const perThread = new SeqNode([initAcc, ktLoop, writeLoop]);

  const locals = new AllocateNode(acc, 'local',
    new AllocateNode(af, 'local',
      new AllocateNode(bf, 'local', perThread)));

  const letChain = new LetStmtNode(tid, ADD(MUL(ty, I(tX)), tx),
    new LetStmtNode(brow, MUL(by, I(BM)),
      new LetStmtNode(bcol, MUL(bx, I(BN)), locals)));

  const gridX = Math.ceil(N / BN);
  const gridY = Math.ceil(M / BM);
  const threadsNest = forT(by, 'blockIdx.y', gridY,
    forT(bx, 'blockIdx.x', gridX,
      forT(ty, 'threadIdx.y', tY,
        forT(tx, 'threadIdx.x', tX, letChain))));

  return new AllocateNode(As, 'shared', new AllocateNode(Bs, 'shared', threadsNest));
}

export function createMatmulRegisterBlockGPUSketch(configs) {
  const idxVar = new SearchVariable('config_index', configs.map((_, i) => i));
  const sketch = new ScheduleSketch('matmul_register_block_gpu', [idxVar],
    (schedule, blockName, target, params) => {
      const bufs = matmulTileDims(schedule.func, blockName);
      if (!bufs) return;
      const cfg = configs[params.config_index];
      if (!cfg) return;
      const body = buildRegisterBlockedMatmul(bufs, cfg);
      schedule.func.body = body;
      if (schedule.func._setChild) schedule.func._setChild('body', body);
      schedule.func.gpuRegisterBlocked = true;
    });
  sketch.configs = configs;
  sketch.enumerate = () => configs.map((_, i) => ({ config_index: i }));
  return sketch;
}
