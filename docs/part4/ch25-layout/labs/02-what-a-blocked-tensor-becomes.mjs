import {
  tensor, nn, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor(Array.from({ length: 16 * 28 * 28 }, (_, i) => ((i % 37) / 37) - 0.5)).reshape([1, 16, 28, 28]);

function must(found, what) {
  if (!found) throw new Error(`this lab no longer reads the IR it was written against: ${what}`);
  return found;
}

manual_seed(0);
const snapshots = new Map();
const entry = compile(new nn.Sequential(new nn.Conv2d(16, 16, 3, { padding: 1, bias: false })), [x], {
  target: CPUTarget(),
  foldWeights: true,
  optimization: { layout: true },
  trace: {
    level: TraceLevel.DEBUG,
    irSnapshot: { afterGraphPasses: true, afterLowering: true },
    sink: (e) => { if (e.type === 'ir_snapshot') snapshots.set(e.label.split(':')[0], e.text); },
  },
});
await entry._ready;
await entry(x);

const graph = must(snapshots.get('afterGraphPasses'), 'no afterGraphPasses snapshot');
const tir = must(snapshots.get('afterLowering'), 'no afterLowering snapshot');
const js = entry.source();

console.log('=== the graph, once the layout pass has run ===');
console.log('');
for (const line of graph.split('\n')) {
  if (/tera\.conv |tera\.layout_transform/.test(line)) console.log('  ' + line.trim().replace(/\{[^}]*\} /, ''));
}
console.log('');
console.log('  every type is still rank 4; the ":1/8" says dimension 1 is stored in blocks of 8');

const lines = tir.split('\n');
const pack = must(lines.find(l => /\/\/ 8.*\] = buf/.test(l)), 'no blocked store in the lowered IR');
console.log('');
console.log('=== the buffer underneath is rank 5 ===');
console.log('');
console.log('  ' + pack.trim());
console.log('');
console.log('  four loop indices, five subscripts: the channel index was split into a quotient and a remainder');

const accAt = lines.findIndex(l => /block conv_acc/.test(l));
must(accAt > 0, 'no conv_acc block in the lowered IR');
const nest = [];
for (let i = accAt - 1; i >= 0 && /^\s*for \S+ in /.test(lines[i]); i--) nest.unshift(lines[i].trim().replace(' {', ''));
must(nest.length >= 8, `expected at least 8 enclosing loops, found ${nest.length}`);
console.log('');
console.log('=== the loop nest the convolution lowers to ===');
console.log('');
for (let i = 0; i < nest.length; i++) {
  const tail = i === nest.length - 1 ? '   <- the channel block, innermost' : '';
  console.log('  ' + '  '.repeat(i) + nest[i] + tail);
}

const product = must(lines.find(l => /\* buf_/.test(l)), 'no multiply in the accumulation body');
console.log('');
console.log('  ' + product.trim());
console.log('');
console.log('  both operands end in that innermost index, so both walk memory one element at a time');

const emitted = must(js.split('\n').find(l => /\| 0\)/.test(l) && /% 8/.test(l)), 'no split channel index in the emitted source');
console.log('');
console.log('=== and in the emitted JavaScript ===');
console.log('');
console.log('  ' + emitted.trim());
