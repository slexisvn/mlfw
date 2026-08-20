import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
}

function sabotage(...victims) {
  const done = new WeakSet();
  return {
    shouldRun(pass) {
      if (victims.includes(pass.name) && !done.has(pass)) {
        done.add(pass);
        pass.run = () => { throw new Error(`deliberate failure inside ${pass.name}`); };
      }
      return true;
    },
  };
}

for (const errorMode of ['strict', 'resilient']) {
  console.log(`=== errorMode: ${errorMode} ===`);
  const events = [];
  try {
    const compiled = compile(build(), [x], {
      target: CPUTarget(),
      errorMode,
      passContext: sabotage('CallInlinerPass', 'cse'),
      trace: { level: TraceLevel.INFO, sink: (e) => { if (e.type === 'error') events.push(e); } },
    });
    await compiled._ready;
    const result = compiled.result();
    console.log(`  compile returned; succeeded: ${result.succeeded}`);
    console.log(`  errors recorded: ${result.errors.length}`);
    for (const e of result.errors) console.log(`    ${e}`);
    console.log(`  functions marked failed: ${[...result.failedFunctions].join(', ') || 'none'}`);
    console.log(`  kernels emitted: ${result.listKernels().length}`);
  } catch (e) {
    console.log(`  compile threw: ${e.message}`);
    console.log(`  error events seen before the throw: ${events.length}`);
  }
  console.log();
}
