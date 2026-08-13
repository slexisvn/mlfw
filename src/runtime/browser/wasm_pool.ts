import type { WasmParallelInfo } from '../../backend/wasm/codegen.js';
import type { NumericTypedArray } from '../../tensor/types/dtype.js';
import type { WasmBufferView, WasmBufferViewCtor, WasmInstance, WasmKernelFn } from '../io.js';

export function runWasmParallel(
  wasmInst: WasmInstance,
  name: string,
  tensorArgs: readonly NumericTypedArray[],
  shapeValues: readonly number[] | null,
  parallel: WasmParallelInfo,
  mathImportNames?: readonly string[],
): Promise<void>;
export async function runWasmParallel(
  wasmInst: WasmInstance,
  name: string,
  tensorArgs: readonly NumericTypedArray[],
  shapeValues: readonly number[] | null,
  parallel: WasmParallelInfo,
): Promise<void> {
  const { exports, memory, bufferOffsets } = wasmInst;
  const fn = exports[name] as WasmKernelFn;
  const offsets = [...bufferOffsets.values()];
  const nBufs = Math.min(offsets.length, tensorArgs.length);

  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (ArrayBuffer.isView(data)) {
      new (data.constructor as WasmBufferViewCtor)(memory.buffer, offsets[i], data.length).set(data);
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
      (data as WasmBufferView).set(new (data.constructor as WasmBufferViewCtor)(memory.buffer, offsets[i], data.length));
    }
  }
}
