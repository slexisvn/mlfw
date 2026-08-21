import {
  compile, CPUTarget, TraceLevel, randn, nn,
} from '../../../../dist/index.node.js';

// Every value in the graph becomes a buffer. This lab counts them, shows the
// one place the lowering rules ask for a scratch buffer in a scope other than
// global, and shows what the block's declared read set leaves out.

async function tir(fn, inputs, opts = {}) {
  const snaps = [];
  const compiled = compile(typeof fn === 'function' ? { forward: fn } : fn, inputs, {
    target: CPUTarget(),
    fusion: { enabled: false },
    ...opts,
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterLowering: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  await compiled(...inputs);
  return { text: snaps[0], src: compiled.source() };
}

console.log('=== a scratch buffer in local scope ===\n');
const arg = await tir((x) => x.argmax(1), [randn([2, 3])]);
console.log(arg.text.split('\n').map((l) => '  ' + l).join('\n'));
console.log('\n  and what the backend does with scope=local:\n');
console.log(arg.src.split('\n').filter((l) => /_argval_|for \(/.test(l)).map((l) => '  ' + l.trim()).join('\n'));

console.log('\n\n=== how many buffers does a model need? ===\n');

class MLP extends nn.Module {
  constructor() {
    super();
    this.fc1 = new nn.Linear(64, 128);
    this.fc2 = new nn.Linear(128, 10);
  }
  forward(a) { return this.fc2.forward(this.fc1.forward(a).relu()); }
}

for (const fusion of [false, true]) {
  const r = await tir(new MLP(), [randn([32, 64])], { fusion: { enabled: fusion } });
  const all = new Set(r.text.match(/buf_\d+/g) || []);
  const bound = new Set((r.text.match(/(buf_\d+) = buffer_map/g) || []).map((s) => s.split(' ')[0]));
  console.log(`  fusion=${String(fusion).padEnd(5)}  buffers: ${String(all.size).padStart(2)}`
    + `   bound to a parameter: ${bound.size}   internal: ${all.size - bound.size}`);
}

console.log('\n\n=== the declared read set is not the read set ===\n');
const red = await tir((x) => x.sum(1), [randn([2, 3])]);
for (const [, name, body] of red.text.matchAll(/block (\w+) \{([\s\S]*?)\n\s*\}/g)) {
  const declared = new Set(((body.match(/reads\(\[(.*?)\]\)/) || [, ''])[1].match(/buf_\d+/g) || []));
  const actual = new Set();
  for (const line of body.split('\n')) {
    const store = line.match(/^\s*(buf_\d+)\[[^\]]*\] = (.*)$/);
    if (!store) continue;
    for (const b of store[2].match(/buf_\d+/g) || []) actual.add(b);
  }
  const missing = [...actual].filter((b) => !declared.has(b));
  console.log(`  ${name.padEnd(15)} declared reads: {${[...declared].join(', ')}}`
    + `  actually read: {${[...actual].join(', ')}}`
    + (missing.length ? `  MISSING: ${missing.join(', ')}` : ''));
}
console.log('\n  An accumulation block reads its own output and does not say so.');
console.log('  Nothing verifies the declaration, and every consumer that has to be');
console.log('  right walks the body instead of trusting it.');
