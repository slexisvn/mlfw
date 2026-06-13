import { cu, checkCU, ATTR_CC_MAJOR, ATTR_CC_MINOR } from './ffi.js';

let _ctx = null;

export function getDevice() {
  if (_ctx) return _ctx;
  checkCU('cuInit', cu.init(0));
  const dev = [0];
  checkCU('cuDeviceGet', cu.deviceGet(dev, 0));
  const major = [0], minor = [0];
  checkCU('cuDeviceGetAttribute', cu.deviceGetAttribute(major, ATTR_CC_MAJOR, dev[0]));
  checkCU('cuDeviceGetAttribute', cu.deviceGetAttribute(minor, ATTR_CC_MINOR, dev[0]));
  const ctx = [null];
  checkCU('cuCtxCreate', cu.ctxCreate(ctx, 0, dev[0]));
  const stream = [null];
  checkCU('cuStreamCreate', cu.streamCreate(stream, 0));
  _ctx = { dev: dev[0], ctx: ctx[0], stream: stream[0], arch: 'sm_' + major[0] + minor[0] };
  process.on('exit', () => { try { cu.ctxDestroy(_ctx.ctx); } catch (_) {} });
  return _ctx;
}
