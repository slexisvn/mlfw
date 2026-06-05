export const MemoryFormat = Object.freeze({
  CONTIGUOUS: 'contiguous',
  CHANNELS_LAST: 'channels_last',
  PRESERVE: 'preserve',
});

const _CHANNELS_LAST_PERM_4D = Object.freeze([0, 2, 3, 1]);
const _CHANNELS_LAST_PERM_5D = Object.freeze([0, 2, 3, 4, 1]);

export function memoryFormatPermutation(ndim, format) {
  if (format === MemoryFormat.CONTIGUOUS || format === MemoryFormat.PRESERVE) return null;
  if (format === MemoryFormat.CHANNELS_LAST) {
    if (ndim === 4) return _CHANNELS_LAST_PERM_4D;
    if (ndim === 5) return _CHANNELS_LAST_PERM_5D;
  }
  return null;
}

export function isChannelsLast(sizes, strides) {
  if (sizes.length < 4) return false;
  const ndim = sizes.length;
  let expected = 1;
  const order = ndim === 4 ? _CHANNELS_LAST_PERM_4D : _CHANNELS_LAST_PERM_5D;
  if (!order) return false;
  for (let i = ndim - 1; i >= 0; i--) {
    const d = order[i];
    if (sizes[d] === 0) return false;
    if (strides[d] !== expected) return false;
    expected *= sizes[d];
  }
  return true;
}
