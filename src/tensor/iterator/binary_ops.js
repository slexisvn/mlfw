import { contiguousLoop, stridedLoop } from './loops.js';

export function binaryLoop(iter, scalarFn) {
  if (iter.isContiguous) {
    _binaryContiguous(iter, scalarFn);
  } else {
    _binaryStrided(iter, scalarFn);
  }
  return iter.output();
}

function _binaryContiguous(iter, fn) {
  const outData = iter.data(0);
  const aData = iter.data(1);
  const bData = iter.data(2);
  const outOff = iter.offsetAt(0);
  const aOff = iter.offsetAt(1);
  const bOff = iter.offsetAt(2);
  const n = iter.numel;
  for (let i = 0; i < n; i++) {
    outData[outOff + i] = fn(aData[aOff + i], bData[bOff + i]);
  }
}

function _binaryStrided(iter, fn) {
  stridedLoop(iter, (data, offsets) => {
    data[0][offsets[0]] = fn(data[1][offsets[1]], data[2][offsets[2]]);
  });
}

export function unaryLoop(iter, scalarFn) {
  if (iter.isContiguous) {
    _unaryContiguous(iter, scalarFn);
  } else {
    _unaryStrided(iter, scalarFn);
  }
  return iter.output();
}

function _unaryContiguous(iter, fn) {
  const outData = iter.data(0);
  const inData = iter.data(1);
  const outOff = iter.offsetAt(0);
  const inOff = iter.offsetAt(1);
  const n = iter.numel;
  for (let i = 0; i < n; i++) {
    outData[outOff + i] = fn(inData[inOff + i]);
  }
}

function _unaryStrided(iter, fn) {
  stridedLoop(iter, (data, offsets) => {
    data[0][offsets[0]] = fn(data[1][offsets[1]]);
  });
}

export function reduceLoop(iter, identity, combineFn, finalizeFn) {
  const inData = iter.data(1);
  const inStrides = iter.stridesAt(1);
  const inOffset = iter.offsetAt(1);
  const outData = iter.data(0);
  const outStrides = iter.stridesAt(0);
  const outOffset = iter.offsetAt(0);

  const inputShape = iter.inputShape();
  const reduceDims = iter.reduceDims;
  const ndim = inputShape.length;

  if (!reduceDims || reduceDims.length === 0) {
    const n = iter.numel;
    let acc = identity;
    const indices = new Int32Array(ndim);
    let off = inOffset;
    for (let i = 0; i < n; i++) {
      acc = combineFn(acc, inData[off]);
      for (let d = ndim - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < inputShape[d]) {
          off += inStrides[d];
          break;
        }
        off -= (inputShape[d] - 1) * inStrides[d];
        indices[d] = 0;
      }
    }
    if (finalizeFn) acc = finalizeFn(acc, n);
    outData[outOffset] = acc;
    return iter.output();
  }

  const dims = new Set(reduceDims.map(d => d < 0 ? ndim + d : d));
  const outerShape = [];
  const outerStridesIn = [];
  const outerStridesOut = [];
  const innerShape = [];
  const innerStridesIn = [];
  const outShape = iter.output().shape;

  let oi = 0;
  for (let d = 0; d < ndim; d++) {
    if (dims.has(d)) {
      innerShape.push(inputShape[d]);
      innerStridesIn.push(inStrides[d]);
    } else {
      outerShape.push(inputShape[d]);
      outerStridesIn.push(inStrides[d]);
      outerStridesOut.push(oi < outStrides.length ? outStrides[oi] : 0);
      oi++;
    }
  }

  const outerNumel = outerShape.reduce((a, b) => a * b, 1);
  const innerNumel = innerShape.reduce((a, b) => a * b, 1);
  const outerNdim = outerShape.length;
  const innerNdim = innerShape.length;

  const outerIndices = new Int32Array(outerNdim);
  let outerInOff = inOffset;
  let outerOutOff = outOffset;

  for (let o = 0; o < outerNumel; o++) {
    let acc = identity;
    const innerIndices = new Int32Array(innerNdim);
    let innerOff = outerInOff;

    for (let i = 0; i < innerNumel; i++) {
      acc = combineFn(acc, inData[innerOff]);
      for (let d = innerNdim - 1; d >= 0; d--) {
        innerIndices[d]++;
        if (innerIndices[d] < innerShape[d]) {
          innerOff += innerStridesIn[d];
          break;
        }
        innerOff -= (innerShape[d] - 1) * innerStridesIn[d];
        innerIndices[d] = 0;
      }
    }

    if (finalizeFn) acc = finalizeFn(acc, innerNumel);
    outData[outerOutOff] = acc;

    for (let d = outerNdim - 1; d >= 0; d--) {
      outerIndices[d]++;
      if (outerIndices[d] < outerShape[d]) {
        outerInOff += outerStridesIn[d];
        outerOutOff += outerStridesOut[d];
        break;
      }
      outerInOff -= (outerShape[d] - 1) * outerStridesIn[d];
      outerOutOff -= (outerShape[d] - 1) * outerStridesOut[d];
      outerIndices[d] = 0;
    }
  }

  return iter.output();
}

export function comparisonLoop(iter, cmpFn) {
  if (iter.isContiguous) {
    _comparisonContiguous(iter, cmpFn);
  } else {
    _comparisonStrided(iter, cmpFn);
  }
  return iter.output();
}

function _comparisonContiguous(iter, fn) {
  const outData = iter.data(0);
  const aData = iter.data(1);
  const bData = iter.data(2);
  const outOff = iter.offsetAt(0);
  const aOff = iter.offsetAt(1);
  const bOff = iter.offsetAt(2);
  const n = iter.numel;
  for (let i = 0; i < n; i++) {
    outData[outOff + i] = fn(aData[aOff + i], bData[bOff + i]) ? 1 : 0;
  }
}

function _comparisonStrided(iter, fn) {
  stridedLoop(iter, (data, offsets) => {
    data[0][offsets[0]] = fn(data[1][offsets[1]], data[2][offsets[2]]) ? 1 : 0;
  });
}
