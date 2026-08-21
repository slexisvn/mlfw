import {
  tensor, Module, compile, trace, printModule, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[1, 2, 3], [4, 5, 6]]);

class Softmax extends Module {
  forward(a) { return a.softmax(1); }
}

console.log('=== traced: what the user wrote ===');
console.log(printModule(await trace((a) => new Softmax().forward(a), [x])));

const stages = new Map();
let decomposed = null;
const compiled = compile(new Softmax(), [x], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.DEBUG,
    irSnapshot: { afterGraphPasses: true },
    sink: (e) => {
      if (e.type === 'pass_detail' && e.passName === 'DecompositionPass') decomposed = e;
      if (e.type === 'pass') stages.set(e.passName, `${e.opCountBefore} -> ${e.opCountAfter}`);
      if (e.type === 'ir_snapshot') stages.set('__ir', e.text);
    },
  },
});
await compiled._ready;

console.log(`=== decomposition report ===`);
console.log(`  ${JSON.stringify(decomposed.decomposed)}  total ${decomposed.totalDecomposed}`);
console.log(`  DecompositionPass: ${stages.get('DecompositionPass')} ops`);

console.log('\n=== after every graph pass ===');
console.log(stages.get('__ir'));

console.log(`kernels emitted: ${compiled.kernels().length}`);
console.log(`output: ${JSON.stringify((await compiled(x)).toArray())}`);
