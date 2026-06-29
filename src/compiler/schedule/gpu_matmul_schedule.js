import {
  analyzePureMatmul, pickFixedConfig, buildRegisterBlockedMatmul,
} from '../autotune/gpu_matmul_sketch.js';
import { collectAllBlockNames } from '../autotune/block_analysis.js';
import { SchedulePolicy } from './rules.js';
import {
  ForNode, ForKind, SeqNode, AllocateNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode,
  MathOpNode, SyncThreadsNode,
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';

const I = (v) => new IntImmNode(v);
const IV = (n) => new VariableNode(n, 'i32');
const ADD = (a, b) => new MathOpNode('+', a, b);
const MUL = (a, b) => new MathOpNode('*', a, b);
const forS = (v, ext, body) => new ForNode(v, I(0), I(ext), ForKind.SERIAL, body);
const forU = (v, ext, body) => new ForNode(v, I(0), I(ext), ForKind.UNROLLED, body);
const forT = (v, tag, ext, body) => new ForNode(v, I(0), I(ext), ForKind.THREAD_BINDING, body, tag);

function stageTileToShared(name, srcBuf, tileRows, tileCols, rowVar, colVar, globalRow, globalCol) {
  const tile = new Buffer(name, [tileRows, tileCols], srcBuf.dtype, 'shared');
  const fill = new BufferStoreNode(tile, [rowVar, colVar],
    new BufferLoadNode(srcBuf, [globalRow, globalCol]));
  return { tile, fill };
}

function buildTiledSharedMatmul(dims, BS, BK) {
  const { A, B, C, M, N, K, transB } = dims;
  const numKTiles = K / BK;

  const by = IV('ts_by'), bx = IV('ts_bx'), ty = IV('ts_ty'), tx = IV('ts_tx');
  const ko = IV('ts_ko'), ki = IV('ts_ki');
  const acc = new Buffer('ts_acc', [1], 'f32', 'local');

  const gRow = ADD(MUL(by, I(BS)), ty);
  const gCol = ADD(MUL(bx, I(BS)), tx);
  const kBase = MUL(ko, I(BK));

  const As = stageTileToShared('ts_As', A, BS, BK, ty, tx, gRow, ADD(kBase, tx));
  const Bs = stageTileToShared('ts_Bs', B, BK, BS, ty, tx,
    transB ? gCol : ADD(kBase, ty),
    transB ? ADD(kBase, ty) : gCol);

  const compute = forU(ki, BK, new BufferStoreNode(acc, [I(0)],
    ADD(new BufferLoadNode(acc, [I(0)]),
        MUL(new BufferLoadNode(As.tile, [ty, ki]), new BufferLoadNode(Bs.tile, [ki, tx])))));

  const ktBody = new SeqNode([As.fill, Bs.fill, new SyncThreadsNode(), compute, new SyncThreadsNode()]);
  const ktLoop = forS(ko, numKTiles, ktBody);

  const init = new BufferStoreNode(acc, [I(0)], new FloatImmNode(0));
  const writeBack = new BufferStoreNode(C, [gRow, gCol], new BufferLoadNode(acc, [I(0)]));
  const perThread = new AllocateNode(acc, 'local', new SeqNode([init, ktLoop, writeBack]));

  return forT(by, 'blockIdx.y', M / BS,
    forT(bx, 'blockIdx.x', N / BS,
      new AllocateNode(As.tile, 'shared',
        new AllocateNode(Bs.tile, 'shared',
          forT(ty, 'threadIdx.y', BS,
            forT(tx, 'threadIdx.x', BS, perThread))))));
}

function pickTiledConfig(target, dims) {
  const { M, N, K } = dims;
  const BS = 16, BK = 16;
  if (BS * BS > (target.maxThreadsPerBlock || 1024)) return null;
  if ((BS * BK + BK * BS) * 4 > (target.sharedMemoryBytes || 49152)) return null;
  if (M % BS !== 0 || N % BS !== 0 || K % BK !== 0) return null;
  if (M < BS || N < BS || K < BK) return null;
  return { BS, BK };
}

export function applyDeterministicGpuConv(schedule, target) {
  const names = collectAllBlockNames(schedule.func.body);
  const convBlocks = names.filter(n => /^q?conv_(init|acc)_/.test(n));
  if (convBlocks.length === 0) return false;
  const policy = new SchedulePolicy(target);
  for (const bn of convBlocks) policy.applyToBlock(schedule, bn);
  return true;
}

export function applyDeterministicGpuMatmul(schedule, target, sCfg = {}) {
  if (!target.isGPU()) return false;
  const plan = analyzePureMatmul(schedule.func);
  if (!plan) return false;
  const dims = plan.dims;

  if (sCfg && sCfg.primitiveMatmul) {
    const tcfg = pickTiledConfig(target, dims);
    if (tcfg) {
      const body = buildTiledSharedMatmul(dims, tcfg.BS, tcfg.BK);
      schedule.func.body = body;
      if (schedule.func._setChild) schedule.func._setChild('body', body);
      schedule.func.gpuRegisterBlocked = true;
      return true;
    }
  }

  const cfg = pickFixedConfig(target, dims);
  if (!cfg) return false;
  const body = buildRegisterBlockedMatmul(dims, cfg);
  schedule.func.body = body;
  if (schedule.func._setChild) schedule.func._setChild('body', body);
  schedule.func.gpuRegisterBlocked = true;
  return true;
}
