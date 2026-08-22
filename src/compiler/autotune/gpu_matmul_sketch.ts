import {
  ForNode, ForKind, SeqNode, AllocateNode, LetStmtNode, IfThenElseNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode,
  MathOpNode, CompareNode, SyncThreadsNode, mathOp,
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { classifyBlock } from '../schedule/rules.js';
import { ScheduleSketch, SearchVariable } from './sketch.js';
import { findBlock, collectAllBlockNames, analyzeBlockStructure } from './block_analysis.js';
import { walk, transform, STOP } from '../ir/ir_visitor.js';
import { cloneTensorIR } from './tune_ir.js';
import { FuncAttr } from '../ir/func_attrs.js';
import type { IRNode } from '../ir/ir_visitor.js';
import type { SketchParams } from './sketch.js';
import type { BlockClassification } from '../schedule/rules.js';
import type { TirNode, PrimFunc } from '../ir/tensor/nodes.js';

type NodeSlots = Record<string, TirNode | TirNode[] | undefined>;
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';

export type MatmulTileDims = {
  A: Buffer; B: Buffer; C: Buffer;
  M: number; N: number; K: number;
  transB?: boolean;
  batch?: number;
};
export type RegisterBlockConfig = { BM: number; BN: number; BK: number; TM: number; TN: number; threads?: number };
export type EpilogueSpec = {
  storeValue: TirNode;
  inputName: string;
  scalarConsts: Map<string, TirNode>;
  iv0: string;
  iv1: string;
  outBuffer: Buffer;
  [key: string]: unknown;
};
export type EpiloguePlan = { reductionBlock: string; dims: MatmulTileDims; epilogue: EpilogueSpec | null };

const I = (v: number): IntImmNode => new IntImmNode(v);
const FZERO = (): FloatImmNode => new FloatImmNode(0);
const IV = (name: string): VariableNode => new VariableNode(name, 'i32');
const ADD = (a: TirNode, b: TirNode): TirNode => mathOp('+', a, b);
const MUL = (a: TirNode, b: TirNode): TirNode => mathOp('*', a, b);
const DIV = (a: TirNode, b: TirNode): TirNode => mathOp('//', a, b);
const MOD = (a: TirNode, b: TirNode): TirNode => mathOp('%', a, b);
const LT = (a: TirNode, b: TirNode): CompareNode => new CompareNode('lt', a, b);
const AND = (a: TirNode, b: TirNode): MathOpNode => new MathOpNode('&&', a, b);

const forS = (v: VariableNode, ext: number, body: TirNode): ForNode => new ForNode(v, I(0), I(ext), ForKind.SERIAL, body);
const forU = (v: VariableNode, ext: number, body: TirNode): ForNode => new ForNode(v, I(0), I(ext), ForKind.UNROLLED, body);
const forT = (v: VariableNode, tag: string, ext: number, body: TirNode): ForNode => new ForNode(v, I(0), I(ext), ForKind.THREAD_BINDING, body, tag);

function pow2Range(min: number, max: number): number[] {
  const out: number[] = [];
  for (let v = 1; v <= max; v *= 2) {
    if (v >= min) out.push(v);
  }
  return out;
}

function plainVars(idxArr: readonly TirNode[]): string[] | null {
  if (!idxArr) return null;
  const out: string[] = [];
  for (const ix of idxArr) {
    if (!ix || ix.type !== 'VariableNode') return null;
    out.push((ix as VariableNode).name);
  }
  return out;
}

function findAccStore(node: TirNode | null | undefined, cName: string): BufferStoreNode | null {
  const stack: (TirNode | null | undefined)[] = [node];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object') continue;
    const st = n as BufferStoreNode;
    if (n.type === 'BufferStoreNode' && st.buffer && st.buffer.name === cName
        && st.value && st.value.type === 'MathOpNode' && (st.value as MathOpNode).op === '+') {
      return st;
    }
    const slots = n as unknown as NodeSlots;
    if (slots.body) stack.push(slots.body as TirNode);
    if (slots.stmts) for (const x of slots.stmts as TirNode[]) stack.push(x);
    if (slots.thenBody) stack.push(slots.thenBody as TirNode);
    if (slots.elseBody) stack.push(slots.elseBody as TirNode);
  }
  return null;
}

function isContiguousRowMajor(buf: Buffer): boolean {
  if (!buf || buf.broadcastDims || (buf.offset && buf.offset !== 0)) return false;
  const sh = buf.shape, st = buf.strides;
  if (!st || st.length !== sh.length) return false;
  let s = 1;
  for (let i = sh.length - 1; i >= 0; i--) {
    if (typeof sh[i] !== 'number' || (sh[i] as number) <= 0 || st[i] !== s) return false;
    s *= sh[i] as number;
  }
  return true;
}

export function matmulTileDims(primFunc: PrimFunc, blockName: string): MatmulTileDims | null {
  const info = classifyBlock(primFunc, blockName);
  if (!info) return null;
  const block = findBlock(primFunc.body, blockName);
  if (!block || block.reads.length < 2 || block.writes.length < 1) return null;
  const C = block.writes[0].buffer;
  if (!C || C.shape.length < 2) return null;
  const R = C.shape.length;

  const store = findAccStore(block.body, C.name);
  if (!store) return null;
  const ci = plainVars(store.indices);
  if (!ci || ci.length !== R) return null;
  const v = store.value as MathOpNode;
  const isCLoad = (x: TirNode | null | undefined): boolean => !!x && x.type === 'BufferLoadNode' && !!(x as BufferLoadNode).buffer && (x as BufferLoadNode).buffer.name === C.name;
  const prod = (isCLoad(v.a) ? v.b : isCLoad(v.b) ? v.a : null) as MathOpNode | null;
  if (!prod || prod.type !== 'MathOpNode' || prod.op !== '*') return null;
  const loads = [prod.a, prod.b] as TirNode[];
  if (!loads.every((l: TirNode) => l && l.type === 'BufferLoadNode' && (l as BufferLoadNode).buffer)) return null;

  const vn = ci[R - 1];
  const rowVars = ci.slice(0, R - 1);
  let A: BufferLoadNode | null = null, vk: string | null = null;
  for (const ld of loads) {
    const idx = plainVars((ld as BufferLoadNode).indices);
    if (!idx || idx.length !== R) continue;
    let leadMatch = true;
    for (let i = 0; i < R - 1; i++) if (idx[i] !== rowVars[i]) { leadMatch = false; break; }
    if (leadMatch) { A = ld as BufferLoadNode; vk = idx[R - 1]; break; }
  }
  if (!A || vk == null) return null;
  const B = (loads[0] === (A as unknown as TirNode) ? loads[1] : loads[0]) as BufferLoadNode;
  const bi = plainVars(B.indices);
  if (!bi) return null;

  const Abuf = A.buffer, Bbuf = B.buffer;
  if (Abuf.shape.length !== R) return null;
  if (Abuf.dtype !== 'f32' || Bbuf.dtype !== 'f32' || C.dtype !== 'f32') return null;
  for (let i = 0; i < R - 1; i++) if (Abuf.shape[i] !== C.shape[i]) return null;
  const N = C.shape[R - 1] as number, K = Abuf.shape[R - 1] as number;
  if (![N, K].every(d => typeof d === 'number' && d > 0)) return null;

  if (bi.length === 2 && Bbuf.shape.length === 2) {
    let transB;
    if (bi[0] === vk && bi[1] === vn) transB = false;
    else if (bi[0] === vn && bi[1] === vk) transB = true;
    else return null;
    if (transB) { if (Bbuf.shape[0] !== N || Bbuf.shape[1] !== K) return null; }
    else { if (Bbuf.shape[0] !== K || Bbuf.shape[1] !== N) return null; }
    let M = 1;
    for (let i = 0; i < R - 1; i++) { const d = C.shape[i]; if (typeof d !== 'number' || d <= 0) return null; M *= d as number; }
    if (R === 2) return { A: Abuf, B: Bbuf, C, M, N, K, transB, batch: 1 };
    if (!isContiguousRowMajor(Abuf) || !isContiguousRowMajor(C)) return null;
    return {
      A: new Buffer(Abuf.name, [M, K], Abuf.dtype, Abuf.scope),
      B: Bbuf,
      C: new Buffer(C.name, [M, N], C.dtype, C.scope),
      M, N, K, transB, batch: 1,
    };
  }

  if (R >= 3 && bi.length === R && Bbuf.shape.length === R) {
    const batchVars = ci.slice(0, R - 2);
    for (let i = 0; i < R - 2; i++) if (bi[i] !== batchVars[i]) return null;
    let transB;
    if (bi[R - 2] === vk && bi[R - 1] === vn) transB = false;
    else if (bi[R - 2] === vn && bi[R - 1] === vk) transB = true;
    else return null;
    for (let i = 0; i < R - 2; i++) if (Bbuf.shape[i] !== C.shape[i]) return null;
    if (transB) { if (Bbuf.shape[R - 2] !== N || Bbuf.shape[R - 1] !== K) return null; }
    else { if (Bbuf.shape[R - 2] !== K || Bbuf.shape[R - 1] !== N) return null; }
    const M = C.shape[R - 2];
    if (typeof M !== 'number' || M <= 0) return null;
    let batch = 1;
    for (let i = 0; i < R - 2; i++) { const d = C.shape[i]; if (typeof d !== 'number' || d <= 0) return null; batch *= d; }
    if (!isContiguousRowMajor(Abuf) || !isContiguousRowMajor(Bbuf) || !isContiguousRowMajor(C)) return null;
    const aV = new Buffer(Abuf.name, [batch, M, K], Abuf.dtype, Abuf.scope);
    const bV = new Buffer(Bbuf.name, transB ? [batch, N, K] : [batch, K, N], Bbuf.dtype, Bbuf.scope);
    const cV = new Buffer(C.name, [batch, M, N], C.dtype, C.scope);
    return { A: aV, B: bV, C: cV, M, N, K, transB, batch };
  }

  return null;
}

function enumerateRegisterBlockConfigs(target: ScheduleTarget, dims: MatmulTileDims | { M: number; N: number; K: number }, maxCandidates = 32): RegisterBlockConfig[] {
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

function goodness(c: RegisterBlockConfig, warp: number): number {
  const reuse = c.TM * c.TN;
  const squareTile = -Math.abs(c.TM - c.TN);
  const squareBlock = -Math.abs(Math.log2(c.BM) - Math.log2(c.BN));
  const occ = (c.threads as number) >= 4 * warp && (c.threads as number) <= 8 * warp ? 1 : 0;
  const shallowK = c.BK === warp / 4 ? 1 : 0;
  return occ * 100 + reuse * 4 + squareTile * 6 + squareBlock * 4 + shallowK;
}

export function pickFixedConfig(target: ScheduleTarget, dims: MatmulTileDims | { M: number; N: number; K: number }): RegisterBlockConfig | null {
  if (!dims) return null;
  const all = enumerateRegisterBlockConfigs(target, dims, 256);
  const preferred = all.find(c => c.BM === 64 && c.BN === 64 && c.BK === 8 && c.TM === 4 && c.TN === 4);
  return preferred || (all.length > 0 ? all[0] : null);
}

function foldEpilogue(ep: EpilogueSpec, accLoad: TirNode, rowC: TirNode, colC: TirNode): TirNode {
  const cloned = cloneTensorIR(ep.storeValue);
  return transform(cloned, (n: IRNode) => {
    const ld = n as BufferLoadNode;
    if (n.type === 'BufferLoadNode' && ld.buffer) {
      if (ld.buffer.name === ep.inputName) return accLoad as IRNode;
      if (ep.scalarConsts.has(ld.buffer.name)) return cloneTensorIR(ep.scalarConsts.get(ld.buffer.name) as TirNode) as IRNode;
      return n;
    }
    if (n.type === 'VariableNode') {
      const vn = n as VariableNode;
      if (vn.name === ep.iv0) return cloneTensorIR(rowC) as IRNode;
      if (vn.name === ep.iv1) return cloneTensorIR(colC) as IRNode;
      return n;
    }
    return n;
  }) as TirNode;
}

export function buildRegisterBlockedMatmul(bufs: MatmulTileDims, params: RegisterBlockConfig, epilogue: EpilogueSpec | null = null): TirNode {
  const { A, B, C, M, N, K, transB } = bufs;
  const batch = bufs.batch || 1;
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
  const k0 = IV('rb_k0'), bz = IV('rb_bz');
  const batched = batch > 1;
  const gIdx = (r: TirNode, c: TirNode): TirNode[] => batched ? [bz, r, c] : [r, c];

  const accIdx = (mi: TirNode, ni: TirNode): TirNode => ADD(MUL(mi, I(TN)), ni);

  const im = IV('rb_im'), inn = IV('rb_in');
  const initAcc = forU(im, TM, forU(inn, TN,
    new BufferStoreNode(acc, [accIdx(im, inn)], FZERO())));

  const la = IV('rb_la'), aidx = IV('rb_aidx');
  const aRow = ADD(brow, DIV(aidx, I(BK)));
  const aCol = ADD(k0, MOD(aidx, I(BK)));
  let aVal: TirNode = new BufferLoadNode(A, gIdx(aRow, aCol));
  if (guardRow || guardK) {
    let cond: TirNode | null = guardRow ? LT(aRow, I(M)) : null;
    if (guardK) cond = cond ? AND(cond, LT(aCol, I(K))) : LT(aCol, I(K));
    aVal = new IfThenElseNode(cond as TirNode, aVal, FZERO());
  }
  let aStore: TirNode = new BufferStoreNode(As, [aidx], aVal);
  if (guardALoad) aStore = new IfThenElseNode(LT(aidx, I(aTotal)), aStore);
  const loadA = forS(la, aLoads,
    new LetStmtNode(aidx, ADD(tid, MUL(la, I(numThreads))), aStore));

  const lb = IV('rb_lb'), bidx = IV('rb_bidx');
  const bRow = ADD(k0, DIV(bidx, I(BN)));
  const bCol = ADD(bcol, MOD(bidx, I(BN)));
  let bVal: TirNode = new BufferLoadNode(B, transB ? gIdx(bCol, bRow) : gIdx(bRow, bCol));
  if (guardK || guardCol) {
    let cond: TirNode | null = guardK ? LT(bRow, I(K)) : null;
    if (guardCol) cond = cond ? AND(cond, LT(bCol, I(N))) : LT(bCol, I(N));
    bVal = new IfThenElseNode(cond as TirNode, bVal, FZERO());
  }
  let bStore: TirNode = new BufferStoreNode(Bs, [bidx], bVal);
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
  const accLoad = new BufferLoadNode(acc, [accIdx(wm, wn)]);
  const outBuf = epilogue ? epilogue.outBuffer : C;
  const storeValue = epilogue ? foldEpilogue(epilogue, accLoad, rowC, colC) : accLoad;
  let writeStore: TirNode = new BufferStoreNode(outBuf, gIdx(rowC, colC), storeValue);
  if (guardRow || guardCol) {
    let cond: TirNode | null = guardRow ? LT(rowC, I(M)) : null;
    if (guardCol) cond = cond ? AND(cond, LT(colC, I(N))) : LT(colC, I(N));
    writeStore = new IfThenElseNode(cond as TirNode, writeStore);
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
  let threadsNest = forT(by, 'blockIdx.y', gridY,
    forT(bx, 'blockIdx.x', gridX,
      forT(ty, 'threadIdx.y', tY,
        forT(tx, 'threadIdx.x', tX, letChain))));
  if (batched) threadsNest = forT(bz, 'blockIdx.z', batch, threadsNest);

  return new AllocateNode(As, 'shared', new AllocateNode(Bs, 'shared', threadsNest));
}

function createMatmulRegisterBlockGPUSketch(configs: readonly RegisterBlockConfig[]): ScheduleSketch {
  const idxVar = new SearchVariable('config_index', configs.map((_, i) => i));
  const sketch = new ScheduleSketch('matmul_register_block_gpu', [idxVar],
    (schedule, blockName, target, params) => {
      const bufs = matmulTileDims(schedule.func, blockName);
      if (!bufs) return;
      const cfg = configs[params.config_index as number];
      if (!cfg) return;
      const body = buildRegisterBlockedMatmul(bufs, cfg);
      schedule.func.body = body;
      if (schedule.func._setChild) schedule.func._setChild('body', body);
      schedule.func.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
    });
  const enriched = sketch as ScheduleSketch & { configs: readonly RegisterBlockConfig[]; enumerate: () => SketchParams[] };
  enriched.configs = configs;
  enriched.enumerate = () => configs.map((_, i) => ({ config_index: i }));
  return enriched;
}

export function analyzePureMatmul(primFunc: PrimFunc): { reductionBlock: string; dims: MatmulTileDims } | null {
  const names = collectAllBlockNames(primFunc.body);
  let reductionBlock = null;
  for (const name of names) {
    const s = analyzeBlockStructure(primFunc, name);
    if (s.hasReduction && s.spatial >= 2 && s.reads >= 2) {
      if (reductionBlock) return null;
      reductionBlock = name;
    }
  }
  if (!reductionBlock) return null;
  const dims = matmulTileDims(primFunc, reductionBlock);
  if (!dims) return null;
  for (const name of names) {
    if (name === reductionBlock) continue;
    const info = classifyBlock(primFunc, name);
    if (!info) return null;
    if (info.hasReduction) return null;
    if (info.readBuffers.length > 0) return null;
    for (const w of info.writeBuffers) if (w !== dims.C.name) return null;
  }
  return { reductionBlock, dims };
}

function collectScalarConstBuffers(body: TirNode | null | undefined): Map<string, TirNode> {
  const consts = new Map<string, TirNode>();
  const writeCount = new Map<string, number>();
  walk(body as IRNode, (n) => {
    if (n.type === 'BufferStoreNode' && n.buffer) {
      const bn = n.buffer.name;
      writeCount.set(bn, (writeCount.get(bn) || 0) + 1);
      if ((!n.indices || n.indices.length === 0) && n.value &&
          (n.value.type === 'FloatImmNode' || n.value.type === 'IntImmNode')) {
        consts.set(bn, n.value);
      }
    }
  });
  for (const bn of [...consts.keys()]) if ((writeCount.get(bn) || 0) !== 1) consts.delete(bn);
  return consts;
}

function findEpilogueStore(node: TirNode | null | undefined, outName: string): BufferStoreNode | null {
  let found = null;
  walk(node as IRNode, (n) => {
    if (n.type === 'BufferStoreNode' && n.buffer && n.buffer.name === outName) { found = n; return STOP; }
  });
  return found;
}

function singleBufferLoad(expr: TirNode | null | undefined, name: string): BufferLoadNode | null {
  let found = null, count = 0;
  walk(expr as IRNode, (n) => { if (n.type === 'BufferLoadNode' && n.buffer && n.buffer.name === name) { count++; found = n; } });
  return count === 1 ? found : null;
}

export function analyzeMatmulEpilogue(primFunc: PrimFunc): EpiloguePlan | null {
  const names = collectAllBlockNames(primFunc.body);
  let reductionBlock = null;
  for (const name of names) {
    const s = analyzeBlockStructure(primFunc, name);
    if (s.hasReduction && s.spatial >= 2 && s.reads >= 2) {
      if (reductionBlock) return null;
      reductionBlock = name;
    }
  }
  if (!reductionBlock) return null;
  const dims = matmulTileDims(primFunc, reductionBlock);
  if (!dims) return null;
  const Cm = dims.C.name;

  const allWrites = new Set();
  for (const name of names) {
    const info = classifyBlock(primFunc, name);
    if (!info) return null;
    for (const w of info.writeBuffers) allWrites.add(w);
  }

  const epilogueBlocks: { name: string; info: BlockClassification }[] = [];
  for (const name of names) {
    if (name === reductionBlock) continue;
    const info = classifyBlock(primFunc, name);
    if (!info || info.hasReduction) return null;
    if (info.readBuffers.length === 0) {
      if (info.writeBuffers.every((w) => w === Cm)) continue;
      return null;
    }
    epilogueBlocks.push({ name, info });
  }

  if (epilogueBlocks.length === 0) return { reductionBlock, dims, epilogue: null };
  if (epilogueBlocks.length !== 1) return null;
  if ((dims.batch || 1) !== 1) return null;

  const ep = epilogueBlocks[0];
  if (ep.info.writeBuffers.length !== 1) return null;
  const outName = ep.info.writeBuffers[0];

  const scalarConsts = collectScalarConstBuffers(primFunc.body);
  let readsCm = 0;
  for (const rb of ep.info.readBuffers) {
    if (rb === Cm) { readsCm++; continue; }
    if (allWrites.has(rb) && !scalarConsts.has(rb)) return null;
  }
  if (readsCm !== 1) return null;

  const block = findBlock(primFunc.body, ep.name);
  if (!block) return null;
  const store = findEpilogueStore(block.body, outName);
  if (!store) return null;
  const iv = plainVars(store.indices);
  if (!iv || iv.length !== 2) return null;
  const cmLoad = singleBufferLoad(store.value, Cm);
  if (!cmLoad) return null;
  const cmIdx = plainVars(cmLoad.indices);
  if (!cmIdx || cmIdx.length !== iv.length || cmIdx.some((v, k) => v !== iv[k])) return null;

  return {
    reductionBlock,
    dims,
    epilogue: { outBuffer: store.buffer, storeValue: store.value, inputName: Cm, iv0: iv[0], iv1: iv[1], scalarConsts },
  };
}

const matmulSketchCache = new WeakMap();

export function richMatmulSketches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): ScheduleSketch[] | null {
  const plan = analyzePureMatmul(primFunc);
  if (!plan) return null;
  let sketch = matmulSketchCache.get(primFunc);
  if (sketch === undefined) {
    const configs = enumerateRegisterBlockConfigs(target, plan.dims);
    sketch = configs.length > 0 ? createMatmulRegisterBlockGPUSketch(configs) : null;
    matmulSketchCache.set(primFunc, sketch);
  }
  if (!sketch) return null;
  if (blockName === plan.reductionBlock) return [sketch];
  return [];
}
