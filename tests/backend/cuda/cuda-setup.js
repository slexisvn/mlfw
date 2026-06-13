async function tryLoadCuda() {
  try {
    const { getDevice } = await import('../../../src/io/node/cuda/device.js');
    const dev = getDevice();
    return { arch: dev.arch };
  } catch (_) {
    return null;
  }
}

export const cudaDeps = await tryLoadCuda();
