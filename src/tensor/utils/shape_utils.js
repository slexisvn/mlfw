export function computeStrides(sizes) {
  const ndim = sizes.length;
  const strides = new Array(ndim);
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= sizes[i];
  }
  return strides;
}

export function computeNumel(sizes) {
  let n = 1;
  for (let i = 0; i < sizes.length; i++) n *= sizes[i];
  return n;
}

export function isContiguous(sizes, strides) {
  const ndim = sizes.length;
  if (ndim === 0) return true;
  let expected = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    if (sizes[i] === 0) return true;
    if (sizes[i] !== 1 && strides[i] !== expected) return false;
    expected *= sizes[i];
  }
  return true;
}

export function storageSize(sizes, strides, offset) {
  const ndim = sizes.length;
  if (ndim === 0) return offset + 1;
  let maxOffset = offset;
  for (let i = 0; i < ndim; i++) {
    if (sizes[i] === 0) return 0;
    maxOffset += (sizes[i] - 1) * Math.abs(strides[i]);
  }
  return maxOffset + 1;
}

export function broadcastShapes(a, b) {
  const aLen = a.length;
  const bLen = b.length;
  const maxLen = Math.max(aLen, bLen);
  const result = new Array(maxLen);
  for (let i = 0; i < maxLen; i++) {
    const da = i < aLen ? a[aLen - 1 - i] : 1;
    const db = i < bLen ? b[bLen - 1 - i] : 1;
    if (da === db) {
      result[maxLen - 1 - i] = da;
    } else if (da === 1) {
      result[maxLen - 1 - i] = db;
    } else if (db === 1) {
      result[maxLen - 1 - i] = da;
    } else {
      return null;
    }
  }
  return result;
}

export function broadcastShapesMulti(shapes) {
  if (shapes.length === 0) return [];
  let result = shapes[0];
  for (let i = 1; i < shapes.length; i++) {
    result = broadcastShapes(result, shapes[i]);
    if (!result) return null;
  }
  return result;
}

export function broadcastStrides(sizes, strides, targetShape) {
  const srcNdim = sizes.length;
  const dstNdim = targetShape.length;
  const dimDiff = dstNdim - srcNdim;
  const result = new Array(dstNdim);
  for (let i = 0; i < dstNdim; i++) {
    const srcIdx = i - dimDiff;
    if (srcIdx < 0 || sizes[srcIdx] === 1) {
      result[i] = 0;
    } else {
      result[i] = strides[srcIdx];
    }
  }
  return result;
}

export function inferReshape(sizes, strides, newSizes) {
  const numel = computeNumel(sizes);
  let inferDim = -1;
  let knownNumel = 1;
  const resolved = new Array(newSizes.length);

  for (let i = 0; i < newSizes.length; i++) {
    if (newSizes[i] === -1) {
      if (inferDim !== -1) return null;
      inferDim = i;
      resolved[i] = -1;
    } else {
      resolved[i] = newSizes[i];
      knownNumel *= newSizes[i];
    }
  }

  if (inferDim !== -1) {
    if (knownNumel === 0) return null;
    resolved[inferDim] = (numel / knownNumel) | 0;
    if (resolved[inferDim] * knownNumel !== numel) return null;
  } else if (computeNumel(resolved) !== numel) {
    return null;
  }

  if (isContiguous(sizes, strides)) {
    return { sizes: resolved, strides: computeStrides(resolved), needsCopy: false };
  }

  return { sizes: resolved, strides: computeStrides(resolved), needsCopy: true };
}

export function flatIndex(indices, strides, offset) {
  let idx = offset;
  for (let i = 0; i < indices.length; i++) idx += indices[i] * strides[i];
  return idx;
}

export function expandShape(sizes, targetShape) {
  const srcNdim = sizes.length;
  const dstNdim = targetShape.length;
  if (srcNdim > dstNdim) return null;
  const dimDiff = dstNdim - srcNdim;
  const result = new Array(dstNdim);
  for (let i = 0; i < dstNdim; i++) {
    const srcIdx = i - dimDiff;
    if (srcIdx < 0) {
      result[i] = targetShape[i];
    } else if (sizes[srcIdx] === 1 || sizes[srcIdx] === targetShape[i]) {
      result[i] = targetShape[i];
    } else if (targetShape[i] === -1) {
      result[i] = sizes[srcIdx];
    } else {
      return null;
    }
  }
  return result;
}
