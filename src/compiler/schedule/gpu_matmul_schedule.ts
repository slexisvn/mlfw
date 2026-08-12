import {
  analyzeMatmulEpilogue, pickFixedConfig, buildRegisterBlockedMatmul,
} from '../autotune/gpu_matmul_sketch.js';
import { collectAllBlockNames } from '../autotune/block_analysis.js';
import { applyImplicitGemmConv } from './conv_implicit_gemm.js';
import type { ConvScheduleConfig } from './conv_implicit_gemm.js';
import { SchedulePolicy } from './rules.js';
import {
  ForNode, ForKind, SeqNode, AllocateNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode,
  MathOpNode, SyncThreadsNode,
} from '../ir/tensor/nodes.js';
import { Buffer } from '../ir/tensor/buffer.js';
import { FuncAttr } from '../ir/func_attrs.js';
import type { TirNode } from '../ir/tensor/nodes.js';
import type { Schedule } from './schedule.js';
import type { TargetFeatures as ScheduleTarget } from '../../backend/target.js';
export type { ScheduleTarget };

export type MatmulDims = {
  A: Buffer; B: Buffer; C: Buffer;
  M: number; N: number; K: number;
  transB?: boolean;
  batch?: number;
};
export type TiledConfig = { BS: number; BK: number };
export type GpuScheduleConfig = ConvScheduleConfig;

const I = (v: number): IntImmNode => new IntImmNode(v);
const IV = (n: string): VariableNode => new VariableNode(n, 'i32');
const ADD = (a: TirNode, b: TirNode): MathOpNode => new MathOpNode('+', a, b);
const MUL = (a: TirNode, b: TirNode): MathOpNode => new MathOpNode('*', a, b);
const forS = (v: VariableNode, ext: number, body: TirNode): ForNode => new ForNode(v, I(0), I(ext), ForKind.SERIAL, body);
const forU = (v: VariableNode, ext: number, body: TirNode): ForNode => new ForNode(v, I(0), I(ext), ForKind.UNROLLED, body);
const forT = (v: VariableNode, tag: string, ext: number, body: TirNode): ForNode => new ForNode(v, I(0), I(ext), ForKind.THREAD_BINDING, body, tag);

function stageTileToShared(name: string, srcBuf: Buffer, tileRows: number, tileCols: number, rowVar: TirNode, colVar: TirNode, globalRow: TirNode, globalCol: TirNode): { tile: Buffer; fill: BufferStoreNode } {
  const tile = new Buffer(name, [tileRows, tileCols], srcBuf.dtype, 'shared');
  const fill = new BufferStoreNode(tile, [rowVar, colVar],
    new BufferLoadNode(srcBuf, [globalRow, globalCol]));
  return { tile, fill };
}

function buildTiledSharedMatmul(dims: MatmulDims, BS: number, BK: number): TirNode {
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

function pickTiledConfig(target: ScheduleTarget, dims: MatmulDims): TiledConfig | null {
  const { M, N, K } = dims;
  const BS = 16, BK = 16;
  if (BS * BS > (target.maxThreadsPerBlock || 1024)) return null;
  if ((BS * BK + BK * BS) * 4 > (target.sharedMemoryBytes || 49152)) return null;
  if (M % BS !== 0 || N % BS !== 0 || K % BK !== 0) return null;
  if (M < BS || N < BS || K < BK) return null;
  return { BS, BK };
}

export function applyDeterministicGpuConv(schedule: Schedule, target: ScheduleTarget): boolean {
  const names = collectAllBlockNames(schedule.func.body);
  const convBlocks = names.filter(n => /^q?conv_(init|acc)_/.test(n));
  if (convBlocks.length === 0) return false;
  const policy = new SchedulePolicy(target);
  for (const bn of convBlocks) policy.applyToBlock(schedule, bn);
  return true;
}

export function applyDeterministicGpuMatmul(schedule: Schedule, target: ScheduleTarget, sCfg: GpuScheduleConfig = {}): boolean {
  if (!target.isGPU()) return false;
  const plan = analyzeMatmulEpilogue(schedule.func);
  if (!plan) return false;
  const dims = plan.dims as MatmulDims;

  if (sCfg && sCfg.primitiveMatmul && (dims.batch || 1) === 1 && !plan.epilogue) {
    const tcfg = pickTiledConfig(target, dims);
    if (tcfg) {
      const body = buildTiledSharedMatmul(dims, tcfg.BS, tcfg.BK);
      schedule.func.body = body;
      if (schedule.func._setChild) schedule.func._setChild('body', body);
      schedule.func.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
      return true;
    }
  }

  const cfg = pickFixedConfig(target, dims);
  if (!cfg) return false;
  const body = (buildRegisterBlockedMatmul as unknown as (bufs: MatmulDims, params: unknown, epilogue: unknown) => TirNode)(dims, cfg, plan.epilogue);
  schedule.func.body = body;
  if (schedule.func._setChild) schedule.func._setChild('body', body);
  schedule.func.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
  return true;
}

export function applyDeterministicGpuSchedule(schedule: Schedule, target: ScheduleTarget, sCfg: GpuScheduleConfig = {}): boolean {
  if (!target.isGPU() || (target.isWebGPU && target.isWebGPU())) return false;
  let handled = applyDeterministicGpuMatmul(schedule, target, sCfg);
  if (!handled) handled = applyImplicitGemmConv(schedule, target, sCfg);
  if (!handled) handled = applyDeterministicGpuConv(schedule, target);
  return handled;
}
