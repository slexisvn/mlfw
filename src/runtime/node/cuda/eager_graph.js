import { cu, checkCU } from './ffi.js';
import { getDevice } from './device.js';
import { setEagerCapturing, isEagerCapturing } from '../../../dispatcher/eager_mode.js';

export const CU_STREAM_CAPTURE_MODE_GLOBAL = 0;
export const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;
export const CU_STREAM_CAPTURE_MODE_RELAXED = 2;

export function isGraphCapturing() { return isEagerCapturing(); }

export function beginEagerCapture(mode = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) {
  const { stream } = getDevice();
  setEagerCapturing(true);
  try {
    checkCU('cuStreamBeginCapture', cu.streamBeginCapture(stream, mode));
  } catch (e) {
    setEagerCapturing(false);
    throw e;
  }
}

export function endEagerCapture() {
  const { stream } = getDevice();
  const graph = [null];
  const endCode = cu.streamEndCapture(stream, graph);
  setEagerCapturing(false);
  checkCU('cuStreamEndCapture', endCode);
  const exec = [null];
  checkCU('cuGraphInstantiateWithFlags', cu.graphInstantiate(exec, graph[0], 0n));
  return { graph: graph[0], exec: exec[0] };
}

export function replay(execGraph) {
  const { stream } = getDevice();
  checkCU('cuGraphLaunch', cu.graphLaunch(execGraph, stream));
}

export function syncStream() {
  const { stream } = getDevice();
  checkCU('cuStreamSynchronize', cu.streamSynchronize(stream));
}

export function destroyEagerGraph(g) {
  if (!g) return;
  if (g.exec) cu.graphExecDestroy(g.exec);
  if (g.graph) cu.graphDestroy(g.graph);
}
