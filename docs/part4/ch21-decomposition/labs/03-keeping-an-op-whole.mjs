import {
  tensor, Module, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[1, 2, 3], [4, 5, 6]]);

class Softmax extends Module { forward(a) { return a.softmax(1); } }

function targetClaiming(...nativeOps) {
  const target = CPUTarget();
  const inherited = target.getAttr.bind(target);
  target.getAttr = (key) => key === 'nativeOps' ? new Set(nativeOps) : inherited(key);
  return target;
}

async function attempt(label, target) {
  console.log(`=== ${label} ===`);
  let decomposed = null;
  try {
    const compiled = compile(new Softmax(), [x], {
      target,
      trace: {
        level: TraceLevel.DEBUG,
        sink: (e) => { if (e.type === 'pass_detail' && e.passName === 'DecompositionPass') decomposed = e.decomposed; },
      },
    });
    await compiled._ready;
    console.log(`  decomposed: ${JSON.stringify(decomposed)}`);
    console.log(`  kernels: ${compiled.kernels().length}`);
  } catch (e) {
    console.log(`  decomposed: ${JSON.stringify(decomposed)}`);
    console.log(`  compile failed: ${e.message}`);
  }
}

await attempt('default target', CPUTarget());
await attempt("target claims softmax is native", targetClaiming('softmax'));
