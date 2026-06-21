async function tryLoadCuda() {
  try {
    const { getDevice } = await import('../../../src/runtime/node/cuda/device.js');
    const dev = getDevice();
    await import('#io/cuda_runtime');
    return { arch: dev.arch };
  } catch (_) {
    return null;
  }
}

export const cudaDeps = await tryLoadCuda();
