import { cu, checkCU } from './ffi.js';
import type { CudaHandle } from './ffi.js';
import { getDevice } from './device.js';
import { setEagerCapturing, isEagerCapturing } from '../../../dispatcher/eager_mode.js';

import type { EagerGraph } from '../../io.js';

export type { EagerGraph };

export const CU_STREAM_CAPTURE_MODE_GLOBAL = 0;
export const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;
export const CU_STREAM_CAPTURE_MODE_RELAXED = 2;

export function isGraphCapturing(): boolean { return isEagerCapturing(); }

export function beginEagerCapture(mode: number = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL): void {
  const { stream } = getDevice();
  setEagerCapturing(true);
  try {
    checkCU('cuStreamBeginCapture', cu.streamBeginCapture(stream, mode));
  } catch (e) {
    setEagerCapturing(false);
    throw e;
  }
}

export function endEagerCapture(): EagerGraph {
  const { stream } = getDevice();
  const graph: (CudaHandle | null)[] = [null];
  const endCode = cu.streamEndCapture(stream, graph);
  setEagerCapturing(false);
  checkCU('cuStreamEndCapture', endCode);
  const exec: (CudaHandle | null)[] = [null];
  checkCU('cuGraphInstantiateWithFlags', cu.graphInstantiate(exec, graph[0], 0n));
  return { graph: graph[0], exec: exec[0] };
}

export function replay(execGraph: CudaHandle | null): void {
  const { stream } = getDevice();
  checkCU('cuGraphLaunch', cu.graphLaunch(execGraph, stream));
}

export function syncStream(): void {
  const { stream } = getDevice();
  checkCU('cuStreamSynchronize', cu.streamSynchronize(stream));
}

export function destroyEagerGraph(g: EagerGraph | null | undefined): void {
  if (!g) return;
  if (g.exec) cu.graphExecDestroy(g.exec);
  if (g.graph) cu.graphDestroy(g.graph);
}
