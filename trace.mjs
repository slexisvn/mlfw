import { tensor, Module, Embedding, LSTM, LSTMCell, Linear, cat } from './src/index.js';
import { _traceCore } from './src/tracing/compile.js';
import { Compiler } from './src/compiler/pipeline/compiler.js';
import { WebGPUTarget } from './src/backend/target.js';
import { writeFileSync, appendFileSync } from 'fs';
const OUT = 'trace.out'; writeFileSync(OUT, '');
const log = s => { appendFileSync(OUT, s + '\n'); process.stdout.write(s + '\n'); };

const V = 600, E = 128, H = 256;
class CB extends Module {
  constructor() { super(); this.embed = new Embedding(V, E); this.encoder = new LSTM(E, H, 1, true); this.dec_cell = new LSTMCell(E + H, H); this.head = new Linear(H, V); }
  forward(q, din) {
    const [enc, st] = this.encoder.forward(this.embed.forward(q));
    const B = q.shape[0], S = enc.shape[1], HH = enc.shape[2];
    let h = st[0].reshape([B, HH]), c = st[1].reshape([B, HH]); const steps = [];
    for (let t = 0; t < din.shape[1]; t++) {
      const e = this.embed.forward(din.select(1, t));
      const w = enc.mul(h.reshape([B, 1, HH])).sum(2).softmax(1).reshape([B, S, 1]);
      const ctx = enc.mul(w).sum(1);
      const r = this.dec_cell.forward(cat([e, ctx], 1), [h, c]); h = r[0]; c = r[1]; steps.push(this.head.forward(h));
    }
    return cat(steps, 0);
  }
}
function mkIn(B, L, off) { const rows = []; for (let b = 0; b < B; b++) rows.push(Array.from({ length: L }, (_, i) => (b * 13 + i * 7 + off) % V)); return tensor(rows, { dtype: 'i32' }); }

log('batch | so kernel | buffer trung gian lon nhat | so buffer > 32KB (chan gop)');
for (const B of [1, 8, 16]) {
  const net = new CB(); net.parameters();
  const q = mkIn(B, 14, 3), din = mkIn(B, 8, 1);
  const traced = _traceCore((a, b) => net.forward(a, b), [q, din], { name: 'CB' });
  const result = new Compiler({ target: WebGPUTarget(), verify: false }).compile(traced.graph);
  const nK = result.listKernels().length;
  const plan = result.module.executionPlan;
  let maxBytes = 0, over = 0;
  if (plan) for (const it of plan.intermediates) { let n = 1; for (const d of it.shape) n *= d; const by = n * 4; if (by > maxBytes) maxBytes = by; if (by > 32768) over++; }
  log(String(B).padEnd(5) + ' | ' + String(nK).padStart(8) + '  | ' + (maxBytes / 1024).toFixed(1).padStart(10) + ' KB        | ' + over);
}
log('=> 32KB la gioi han gop cua partitioner (PARTITION_BUFFER_LIMIT). batch lon -> buffer > 32KB -> khong gop -> kernel no.');
