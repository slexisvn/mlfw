import type { ProgressSink } from '../io.js';

let _lastLength = 0;
let _open = false;

export const progress: ProgressSink = {
  update(line) {
    const pad = Math.max(0, _lastLength - line.length);
    process.stdout.write('\r' + line + ' '.repeat(pad));
    _lastLength = line.length;
    _open = true;
  },
  finish() {
    if (_open) process.stdout.write('\n');
    _lastLength = 0;
    _open = false;
  },
};
