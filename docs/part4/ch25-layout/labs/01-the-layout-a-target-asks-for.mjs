import {
  tensor, nn, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor(Array.from({ length: 16 * 28 * 28 }, (_, i) => ((i % 37) / 37) - 0.5)).reshape([1, 16, 28, 28]);

function must(condition, message) {
  if (!condition) throw new Error(`this lab no longer reads the IR it was written against: ${message}`);
}

function build(depth, bias) {
  manual_seed(0);
  const layers = [];
  for (let i = 0; i < depth; i++) {
    layers.push(new nn.Conv2d(16, 16, 3, { padding: 1, bias }));
    layers.push(new nn.ReLU());
  }
  return new nn.Sequential(...layers);
}

function readGraph(text, depth) {
  const convs = text.split('\n').filter(line => line.includes('tera.conv '));
  must(convs.length === depth, `expected ${depth} tera.conv lines, found ${convs.length}`);
  let both = 0, kernelOnly = 0;
  for (const line of convs) {
    const signature = line.match(/:\s*\(([^)]*)\)\s*->/);
    must(signature, `no operand type list in ${line.trim().slice(0, 60)}`);
    const blocked = (signature[1].match(/:1\/8>/g) || []).length;
    if (blocked === 2) both++;
    else if (blocked === 1) kernelOnly++;
  }
  return { transforms: (text.match(/tera\.layout_transform/g) || []).length, both, kernelOnly };
}

async function compiled(depth, bias, layout) {
  let graph = null;
  const entry = compile(build(depth, bias), [x], {
    target: CPUTarget(),
    foldWeights: true,
    optimization: { layout },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot' && e.label === 'afterGraphPasses') graph = e.text; },
    },
  });
  await entry._ready;
  for (let i = 0; i < 8; i++) await entry(x);
  return { entry, graph };
}

async function best(fn, reps) {
  const times = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  return Math.min(...times);
}

async function race(off, on) {
  const a = [], b = [];
  for (let r = 0; r < 4; r++) {
    a.push(await best(() => off(x), 4));
    b.push(await best(() => on(x), 4));
  }
  const offMs = Math.min(...a), onMs = Math.min(...b);
  return { offMs, onMs, spread: Math.max(Math.max(...a) - offMs, Math.max(...b) - onMs) };
}

const HEADER = '  depth  transforms  convs blocked on both operands  on the kernel only    off      on   ratio';

async function row(depth, bias) {
  const plain = await compiled(depth, bias, false);
  const laid = await compiled(depth, bias, true);
  const g = readGraph(laid.graph, depth);
  const { offMs, onMs, spread } = await race(plain.entry, laid.entry);
  console.log(
    `  ${String(depth).padStart(5)}  ${String(g.transforms).padStart(10)}`
    + `  ${String(g.both).padStart(29)}  ${String(g.kernelOnly).padStart(18)}`
    + `  ${offMs.toFixed(2).padStart(5)}  ${onMs.toFixed(2).padStart(5)}  ${(offMs / onMs).toFixed(2)}x`
    + `   (worst of 4 rounds +${spread.toFixed(2)} ms)`);
}

const target = CPUTarget();
console.log('=== what the shipped CPU target asks for ===');
console.log('');
console.log(`  layoutAwareOps       {${[...target.layoutAwareOps].join(', ')}}`);
const spec = target.preferredConvLayout;
console.log(`  preferredConvLayout  order [${spec.order.join(', ')}], dimension ${spec.block.dim} split by ${spec.block.factor}   (NCHW8c)`);
console.log('');
console.log('=== a chain of 3x3 convolutions over 1x16x28x28 ===');
console.log('');
console.log(HEADER);
for (const depth of [1, 2, 3]) await row(depth, false);
console.log('');
console.log('=== the same chain with a bias on every convolution ===');
console.log('');
console.log(HEADER);
await row(3, true);
