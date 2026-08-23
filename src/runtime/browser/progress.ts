import type { ProgressSink } from '../io.js';

export const progress: ProgressSink = {
  update(line) { console.log(line); },
  finish() {},
};
