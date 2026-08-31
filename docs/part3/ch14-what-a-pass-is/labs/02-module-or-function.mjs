import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

const VERDICT = ['UNCHANGED', 'CHANGED', 'FAILED'];
const declared = new Map();
const calls = [];
const instrumented = new WeakSet();

const compiled = compile(model, [x], {
  target: CPUTarget(),
  passContext: {
    shouldRun(pass) {
      if (!Array.isArray(pass.requiredAnalyses)) return true;
      if (!instrumented.has(pass)) {
        instrumented.add(pass);
        declared.set(pass.name, {
          className: pass.constructor.name,
          optLevel: pass.optLevel,
          requires: pass.requiredAnalyses.map(a => a.name),
          preserves: [...pass.preservedAnalyses].map(a => typeof a === 'string' ? a : a.name),
        });
        const inner = pass.run.bind(pass);
        pass.run = (target, analyses) => {
          const result = inner(target, analyses);
          const kind = typeof target.numOps === 'function' ? 'function' : 'module';
          calls.push(`${pass.name.padEnd(23)} run(${kind} '${target.name}') -> ${VERDICT[result]}`);
          return result;
        };
      }
      return true;
    },
  },
});
await compiled._ready;

console.log('=== what the pass manager hands each pass ===');
for (const line of calls) console.log(line);

console.log('\n=== what each pass declares about itself ===');
console.log('pass                    class                       opt  requires        preserves');
for (const [name, d] of declared) {
  console.log(
    `${name.padEnd(23)} ${d.className.padEnd(27)} ${String(d.optLevel).padEnd(4)} ` +
    `${(d.requires.join(',') || '-').padEnd(15)} ${d.preserves.join(',') || '-'}`
  );
}
