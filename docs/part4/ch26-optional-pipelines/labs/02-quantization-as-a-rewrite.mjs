import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor(Array.from({ length: 8 * 32 }, (_, i) => ((i % 31) / 31) - 0.5)).reshape([8, 32]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(32, 64), new ReLU(), new Linear(64, 16));
}

async function study(label, quantization, foldWeights = false) {
  let ir = null;
  const compiled = compile(build(), [x], {
    target: CPUTarget(),
    quantization,
    foldWeights,
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') ir = e.text; },
    },
  });
  await compiled._ready;
  const out = (await compiled(x)).toArray().flat();
  const ops = (ir.match(/= (\w+)\(/g) || []).map(s => s.slice(2, -1));
  console.log(`=== ${label} ===`);
  console.log(`  ${ops.length} operations: ${ops.join(', ')}`);
  return out;
}

const quant = { enabled: true, quantizableOps: new Set(['dot']) };

const reference = await study('float32', undefined);
const RUNS = [
  ['int8, default activation range', quant, false],
  ['int8, calibration', { ...quant, calibrationData: [[x]] }, false],
  ['int8, folded weights', quant, true],
  ['int8, folded weights + calibration', { ...quant, calibrationData: [[x]] }, true],
];

console.log();
for (const [label, quantization, foldWeights] of RUNS) {
  const out = await study(label, quantization, foldWeights);
  let err = 0, mag = 0;
  for (let i = 0; i < reference.length; i++) {
    err += Math.abs(out[i] - reference[i]);
    mag += Math.abs(reference[i]);
  }
  console.log(`  relative error against float32: ${(100 * err / mag).toFixed(2)}%`);
  console.log(`  first output element: ${reference[0]} -> ${out[0]}\n`);
}
