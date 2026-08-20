import {
  tensor, Module, nn, compile, trace, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[1, 2, 3], [4, 5, 6]]);
const layerNorm = new nn.LayerNorm(3);
const elu = new nn.ELU();

const CASES = {
  'sigmoid':     (a) => a.sigmoid(),
  'silu':        (a) => a.silu(),
  'gelu':        (a) => a.gelu(),
  'elu':         (a) => elu.forward(a),
  'softmax':     (a) => a.softmax(1),
  'log_softmax': (a) => a.log_softmax(1),
  'layer_norm':  (a) => layerNorm.forward(a),
  'tanh':        (a) => a.tanh(),
};

console.log('op            traced  after decomposition  after all passes  kernels');
for (const [name, fn] of Object.entries(CASES)) {
  class Case extends Module { forward(a) { return fn(a); } }

  const traced = await trace((a) => fn(a), [x]);
  const tracedOps = [...traced.functions().next().value.ops()].length;

  let afterDecomp = null, afterAll = null;
  const compiled = compile(new Case(), [x], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel.DEBUG,
      sink: (e) => {
        if (e.type !== 'pass') return;
        if (e.passName === 'DecompositionPass') afterDecomp = e.opCountAfter;
        afterAll = e.opCountAfter;
      },
    },
  });
  await compiled._ready;

  console.log(
    `${name.padEnd(13)} ${String(tracedOps).padStart(6)}  ${String(afterDecomp).padStart(19)}` +
    `  ${String(afterAll).padStart(16)}  ${String(compiled.kernels().length).padStart(7)}`
  );
}
