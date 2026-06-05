export function contiguousLoop(iter, fn) {
  const numTensors = iter.numTensors;
  const numel = iter.numel;
  const data = new Array(numTensors);
  const offsets = new Array(numTensors);
  for (let t = 0; t < numTensors; t++) {
    data[t] = iter.data(t);
    offsets[t] = iter.offsetAt(t);
  }

  for (let i = 0; i < numel; i++) {
    fn(data, offsets, i);
    for (let t = 0; t < numTensors; t++) offsets[t]++;
  }
}

export function stridedLoop(iter, fn) {
  const numTensors = iter.numTensors;
  const ndim = iter.ndim;
  const numel = iter.numel;
  const shape = iter.shape;

  if (numel === 0) return;
  if (ndim === 0) {
    const data = new Array(numTensors);
    const offsets = new Array(numTensors);
    for (let t = 0; t < numTensors; t++) {
      data[t] = iter.data(t);
      offsets[t] = iter.offsetAt(t);
    }
    fn(data, offsets, 0);
    return;
  }

  const allStrides = new Array(numTensors);
  for (let t = 0; t < numTensors; t++) {
    allStrides[t] = iter.stridesAt(t);
  }

  const data = new Array(numTensors);
  const offsets = new Int32Array(numTensors);
  for (let t = 0; t < numTensors; t++) {
    data[t] = iter.data(t);
    offsets[t] = iter.offsetAt(t);
  }

  const indices = new Int32Array(ndim);

  for (let i = 0; i < numel; i++) {
    fn(data, offsets, i);

    for (let d = ndim - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < shape[d]) {
        for (let t = 0; t < numTensors; t++) {
          offsets[t] += allStrides[t][d];
        }
        break;
      }
      for (let t = 0; t < numTensors; t++) {
        offsets[t] -= (shape[d] - 1) * allStrides[t][d];
      }
      indices[d] = 0;
    }
  }
}

export function iterLoop(iter, fn) {
  if (iter.isContiguous) {
    contiguousLoop(iter, fn);
  } else {
    stridedLoop(iter, fn);
  }
}
