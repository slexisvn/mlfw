import koffi from 'koffi';
import { readdirSync, existsSync } from 'fs';
import { join } from 'path';
import { cu } from './ffi.js';
import { getDevice } from './device.js';
import { setDevice } from './runtime_api.js';
import { acquire, release } from './memory.js';
import { isEagerCapturing } from '../../../dispatcher/eager_mode.js';

function d2d(dst, src, bytes) {
  if (isEagerCapturing()) cu.memcpyDtoDAsync(dst, src, bytes, getDevice().stream);
  else cu.memcpyDtoD(dst, src, bytes);
}
function memZero(dst, bytes) {
  if (isEagerCapturing()) cu.memsetD8Async(dst, 0, bytes, getDevice().stream);
  else cu.memsetD8(dst, 0, bytes);
}

function resolveCudnn() {
  const root = 'C:/Program Files/NVIDIA/CUDNN';
  if (existsSync(root)) {
    for (const ver of readdirSync(root).sort().reverse()) {
      const binBase = join(root, ver, 'bin');
      if (!existsSync(binBase)) continue;
      for (const cd of readdirSync(binBase).filter(d => d.startsWith('12.')).sort().reverse()) {
        const dir = join(binBase, cd, 'x64');
        if (existsSync(join(dir, 'cudnn64_9.dll'))) {
          if (!process.env.PATH.includes(dir)) process.env.PATH = dir + ';' + process.env.PATH;
          return join(dir, 'cudnn64_9.dll');
        }
      }
    }
  }
  return 'cudnn64_9.dll';
}

let c, _inited = null;
function ensure() {
  if (_inited !== null) return _inited;
  try {
    const dnn = koffi.load(resolveCudnn());
    c = {
      create: dnn.func('int cudnnCreate(_Out_ void **h)'),
      setStream: dnn.func('int cudnnSetStream(void *h, void *stream)'),
      getErrorString: dnn.func('str cudnnGetErrorString(int status)'),
      createRNNDesc: dnn.func('int cudnnCreateRNNDescriptor(_Out_ void **d)'),
      setRNNDesc: dnn.func('int cudnnSetRNNDescriptor_v8(void *d, int algo, int cellMode, int biasMode, int dirMode, int inputMode, int dataType, int mathPrec, int mathType, int inputSize, int hiddenSize, int projSize, int numLayers, void *dropoutDesc, uint auxFlags)'),
      createRNNDataDesc: dnn.func('int cudnnCreateRNNDataDescriptor(_Out_ void **d)'),
      setRNNDataDesc: dnn.func('int cudnnSetRNNDataDescriptor(void *d, int dataType, int layout, int maxSeqLength, int batchSize, int vectorSize, void *seqLengthArray, void *paddingFill)'),
      createTensorDesc: dnn.func('int cudnnCreateTensorDescriptor(_Out_ void **d)'),
      setTensorNdDesc: dnn.func('int cudnnSetTensorNdDescriptor(void *d, int dataType, int nbDims, void *dimA, void *strideA)'),
      createDropoutDesc: dnn.func('int cudnnCreateDropoutDescriptor(_Out_ void **d)'),
      setDropoutDesc: dnn.func('int cudnnSetDropoutDescriptor(void *d, void *h, float dropout, uint64 states, size_t stateSize, uint64 seed)'),
      dropoutGetStatesSize: dnn.func('int cudnnDropoutGetStatesSize(void *h, _Out_ size_t *sz)'),
      getRNNWeightSpaceSize: dnn.func('int cudnnGetRNNWeightSpaceSize(void *h, void *d, _Out_ size_t *sz)'),
      getRNNTempSpaceSizes: dnn.func('int cudnnGetRNNTempSpaceSizes(void *h, void *d, int fwdMode, void *xDesc, _Out_ size_t *work, _Out_ size_t *reserve)'),
      getRNNWeightParams: dnn.func('int cudnnGetRNNWeightParams(void *h, void *d, int pseudoLayer, size_t wss, uint64 weightSpace, int linLayerID, void *mDesc, _Out_ uint64 *mAddr, void *bDesc, _Out_ uint64 *bAddr)'),
      rnnForward: dnn.func('int cudnnRNNForward(void *h, void *d, int fwdMode, uint64 devSeqLengths, void *xDesc, uint64 x, void *yDesc, uint64 y, void *hDesc, uint64 hx, uint64 hy, void *cDesc, uint64 cx, uint64 cy, size_t wss, uint64 weightSpace, size_t workSize, uint64 workSpace, size_t reserveSize, uint64 reserveSpace)'),
      rnnBackwardData: dnn.func('int cudnnRNNBackwardData_v8(void *h, void *d, uint64 devSeqLengths, void *yDesc, uint64 y, uint64 dy, void *xDesc, uint64 dx, void *hDesc, uint64 hx, uint64 dhy, uint64 dhx, void *cDesc, uint64 cx, uint64 dcy, uint64 dcx, size_t wss, uint64 weightSpace, size_t workSize, uint64 workSpace, size_t reserveSize, uint64 reserveSpace)'),
      rnnBackwardWeights: dnn.func('int cudnnRNNBackwardWeights_v8(void *h, void *d, int addGrad, uint64 devSeqLengths, void *xDesc, uint64 x, void *hDesc, uint64 hx, void *yDesc, uint64 y, size_t wss, uint64 dweightSpace, size_t workSize, uint64 workSpace, size_t reserveSize, uint64 reserveSpace)'),
    };
    _inited = true;
  } catch (e) {
    _inited = false;
  }
  return _inited;
}
export function cudnnAvailable() { return ensure(); }

const FLOAT = 0, ALGO_STANDARD = 0, LSTM = 2, DOUBLE_BIAS = 2, UNIDIR = 0, LINEAR_INPUT = 0, ALLOW_CONVERSION = 2;
const LAYOUT_SEQ_MAJOR_UNPACKED = 0;
const FWD_TRAINING = 1, FWD_INFERENCE = 0, WGRAD_ADD = 0;

function ck(label, status) {
  if (status !== 0) throw new Error('cuDNN ' + label + ' failed: ' + status + ' (' + c.getErrorString(status) + ')');
}
const o2 = () => [null];

let _handle = null, _dropout = null;
function handle() {
  if (!ensure()) throw new Error('cuDNN not available');
  const dev = getDevice();
  setDevice();
  if (!_handle) {
    const h = o2(); ck('create', c.create(h)); _handle = h[0];
    ck('setStream', c.setStream(_handle, dev.stream));
  }
  return _handle;
}
function dropoutDesc() {
  if (!_dropout) {
    const h = handle();
    const d = o2(); ck('createDropout', c.createDropoutDesc(d));
    const sz = [0n]; ck('dropStatesSize', c.dropoutGetStatesSize(h, sz));
    const states = acquire(Math.max(Number(sz[0]), 1));
    ck('setDropout', c.setDropoutDesc(d[0], h, 0.0, states, sz[0], 0n));
    _dropout = d[0];
  }
  return _dropout;
}
function tensorDesc(dims, strides) {
  const d = o2(); ck('createTensor', c.createTensorDesc(d));
  ck('setTensorNd', c.setTensorNdDesc(d[0], FLOAT, dims.length, new Int32Array(dims), new Int32Array(strides)));
  return d[0];
}
function dataDesc(seqLen, batch, vec, seqArr) {
  const d = o2(); ck('createRNNData', c.createRNNDataDesc(d));
  ck('setRNNData', c.setRNNDataDesc(d[0], FLOAT, LAYOUT_SEQ_MAJOR_UNPACKED, seqLen, batch, vec, seqArr, null));
  return d[0];
}

const _descCache = new Map();
function plan({ inputSize, hiddenSize, seqLen, batch, numLayers }) {
  const key = `${inputSize}_${hiddenSize}_${seqLen}_${batch}_${numLayers}`;
  let p = _descCache.get(key);
  if (p) return p;
  const h = handle();
  const rnnDesc = o2(); ck('createRNN', c.createRNNDesc(rnnDesc));
  ck('setRNN', c.setRNNDesc(rnnDesc[0], ALGO_STANDARD, LSTM, DOUBLE_BIAS, UNIDIR, LINEAR_INPUT, FLOAT, FLOAT, ALLOW_CONVERSION, inputSize, hiddenSize, hiddenSize, numLayers, dropoutDesc(), 0));
  const rd = rnnDesc[0];
  const seqArr = new Int32Array(batch).fill(seqLen);
  const xDesc = dataDesc(seqLen, batch, inputSize, seqArr);
  const yDesc = dataDesc(seqLen, batch, hiddenSize, seqArr);
  const hDesc = tensorDesc([numLayers, batch, hiddenSize], [batch * hiddenSize, hiddenSize, 1]);
  const wssP = [0n]; ck('wss', c.getRNNWeightSpaceSize(h, rd, wssP));
  const workP = [0n], resvP = [0n];
  ck('temp', c.getRNNTempSpaceSizes(h, rd, FWD_TRAINING, xDesc, workP, resvP));
  const devSeq = acquire(batch * 4);
  cu.memcpyHtoD(devSeq, seqArr, batch * 4);
  p = { rd, xDesc, yDesc, hDesc, devSeq, wss: wssP[0], wssN: Number(wssP[0]),
    workSize: workP[0], workN: Math.max(Number(workP[0]), 1), reserveSize: resvP[0], reserveN: Math.max(Number(resvP[0]), 1),
    inputSize, hiddenSize, seqLen, batch, numLayers };
  _descCache.set(key, p);
  return p;
}

function packWeights(p, weightSpace, layerDevs, download) {
  const h = handle();
  const mDesc = o2(), bDesc = o2();
  ck('cwm', c.createTensorDesc(mDesc)); ck('cwb', c.createTensorDesc(bDesc));
  const { numLayers, hiddenSize, inputSize, wss } = p;
  for (let l = 0; l < numLayers; l++) {
    const inF = l === 0 ? inputSize : hiddenSize;
    const ld = layerDevs[l];
    for (let id = 0; id < 8; id++) {
      const isInput = id < 4, gate = id % 4;
      const mAddr = [0n], bAddr = [0n];
      ck('wparam', c.getRNNWeightParams(h, p.rd, l, wss, weightSpace, id, mDesc[0], mAddr, bDesc[0], bAddr));
      const wLen = isInput ? hiddenSize * inF : hiddenSize * hiddenSize;
      const wBytes = wLen * 4, bBytes = hiddenSize * 4;
      const wSrc = BigInt(isInput ? ld.x2hW : ld.h2hW) + BigInt(gate * wBytes);
      const bSrc = BigInt(isInput ? ld.x2hB : ld.h2hB) + BigInt(gate * bBytes);
      if (download) {
        d2d(wSrc, mAddr[0], wBytes);
        d2d(bSrc, bAddr[0], bBytes);
      } else {
        d2d(mAddr[0], wSrc, wBytes);
        d2d(bAddr[0], bSrc, bBytes);
      }
    }
  }
}

export function cudnnLSTMForward(xDev, layerDevs, opts, hxDev, cxDev, yDev, hyDev, cyDev, training = true) {
  const h = handle();
  const p = plan({ ...opts, numLayers: layerDevs.length });
  const weightSpace = acquire(p.wssN);
  packWeights(p, weightSpace, layerDevs, false);
  const workSpace = acquire(p.workN);
  const reserveSpace = training ? acquire(p.reserveN) : 0n;
  ck('forward', c.rnnForward(h, p.rd, training ? FWD_TRAINING : FWD_INFERENCE, p.devSeq, p.xDesc, xDev, p.yDesc, yDev,
    p.hDesc, hxDev || 0n, hyDev, p.hDesc, cxDev || 0n, cyDev, p.wss, weightSpace, p.workSize, workSpace,
    training ? p.reserveSize : 0n, reserveSpace));
  return { p, weightSpace, workSpace, reserveSpace, training };
}

export function releaseLSTMForward(ctx) {
  if (!ctx) return;
  release(ctx.weightSpace, ctx.p.wssN);
  release(ctx.workSpace, ctx.p.workN);
  if (ctx.reserveSpace && ctx.reserveSpace !== 0n) release(ctx.reserveSpace, ctx.p.reserveN);
}

export function cudnnLSTMBackward(ctx, xDev, yDev, hxDev, cxDev, dyDev, dhyDev, dcyDev, dxDev, dhxDev, dcxDev, gradLayerDevs) {
  const { p, weightSpace, workSpace, reserveSpace } = ctx;
  const h = handle();
  const dweight = acquire(p.wssN);
  memZero(dweight, p.wssN);
  ck('backwardData', c.rnnBackwardData(h, p.rd, p.devSeq, p.yDesc, yDev, dyDev, p.xDesc, dxDev,
    p.hDesc, hxDev || 0n, dhyDev || 0n, dhxDev || 0n, p.hDesc, cxDev || 0n, dcyDev || 0n, dcxDev || 0n,
    p.wss, weightSpace, p.workSize, workSpace, p.reserveSize, reserveSpace));
  ck('backwardWeights', c.rnnBackwardWeights(h, p.rd, WGRAD_ADD, p.devSeq, p.xDesc, xDev, p.hDesc, hxDev || 0n, p.yDesc, yDev,
    p.wss, dweight, p.workSize, workSpace, p.reserveSize, reserveSpace));
  packWeights(p, dweight, gradLayerDevs, true);
  release(weightSpace, p.wssN);
  release(workSpace, p.workN);
  release(reserveSpace, p.reserveN);
  release(dweight, p.wssN);
}
