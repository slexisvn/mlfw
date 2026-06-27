export function flattenRowMajorIndex(buffer, indices, emitLeaf, computeDynamicStride, skipZero = false) {
  if (indices.length === 0) return '0';
  if (indices.length === 1) return emitLeaf(indices[0]);
  const parts = [];
  for (let i = 0; i < indices.length; i++) {
    const idx = emitLeaf(indices[i]);
    if (skipZero && idx === '0') continue;
    const stride = buffer.strides[i];
    if (stride === 1) {
      parts.push(idx);
    } else if (typeof stride === 'number' && stride >= 0) {
      parts.push(`${idx} * ${stride}`);
    } else {
      parts.push(`${idx} * ${computeDynamicStride(buffer, i)}`);
    }
  }
  return parts.length === 0 ? '0' : parts.join(' + ');
}
