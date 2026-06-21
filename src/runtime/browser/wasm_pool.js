export async function runWasmParallel(wasmInst, name, tensorArgs, shapeValues, parallel) {
  const { exports, memory, bufferOffsets } = wasmInst;
  const fn = exports[name];
  const offsets = [...bufferOffsets.values()];
  const nBufs = Math.min(offsets.length, tensorArgs.length);

  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (ArrayBuffer.isView(data)) {
      new data.constructor(memory.buffer, offsets[i], data.length).set(data);
    }
  }

  const callArgs = offsets.slice(0, nBufs);
  if (shapeValues) {
    for (const v of shapeValues) callArgs.push(v);
  }
  callArgs.push(0, parallel.extent);
  fn(...callArgs);

  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (ArrayBuffer.isView(data)) {
      data.set(new data.constructor(memory.buffer, offsets[i], data.length));
    }
  }
}
