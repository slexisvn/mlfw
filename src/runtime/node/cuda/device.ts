import { cu, checkCU, ATTR_CC_MAJOR, ATTR_CC_MINOR } from './ffi.js';
import type { CudaHandle } from './ffi.js';

export type CudaDevice = {
  dev: number;
  ctx: CudaHandle | null;
  stream: CudaHandle | null;
  arch: string;
  totalMem: number;
};

let _ctx: CudaDevice | null = null;

export function getDevice(): CudaDevice {
  if (_ctx) {
    checkCU('cuCtxSetCurrent', cu.ctxSetCurrent(_ctx.ctx));
    return _ctx;
  }
  checkCU('cuInit', cu.init(0));
  const dev = [0];
  checkCU('cuDeviceGet', cu.deviceGet(dev, 0));
  const major = [0], minor = [0];
  checkCU('cuDeviceGetAttribute', cu.deviceGetAttribute(major, ATTR_CC_MAJOR, dev[0]));
  checkCU('cuDeviceGetAttribute', cu.deviceGetAttribute(minor, ATTR_CC_MINOR, dev[0]));
  const ctx: (CudaHandle | null)[] = [null];
  checkCU('cuDevicePrimaryCtxRetain', cu.primaryCtxRetain(ctx, dev[0]));
  checkCU('cuCtxSetCurrent', cu.ctxSetCurrent(ctx[0]));
  const stream: (CudaHandle | null)[] = [null];
  checkCU('cuStreamCreate', cu.streamCreate(stream, 0));
  const free = [0n], total = [0n];
  checkCU('cuMemGetInfo', cu.memGetInfo(free, total));
  _ctx = { dev: dev[0], ctx: ctx[0], stream: stream[0], arch: 'sm_' + major[0] + minor[0], totalMem: Number(total[0]) };
  process.on('exit', () => { try { cu.primaryCtxRelease(_ctx!.dev); } catch (_) {} });
  return _ctx;
}
