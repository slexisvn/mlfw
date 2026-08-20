import {
  tensor, Module, compile, trace, printModule, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[1, 2], [3, 4]]);

const spellings = {
  'a * a':                         (a) => a.mul(a),
  'transpose(transpose(a)) * a':   (a) => a.transpose(1, 0).transpose(1, 0).mul(a),
  '(a + 0) * (a * 1)':             (a) => a.add(0).mul(a.mul(1)),
  'reshape(reshape(a)) * (a - 0)': (a) => a.reshape([4]).reshape([2, 2]).mul(a.sub(0)),
};

const canonical = new Map();

for (const [label, fn] of Object.entries(spellings)) {
  class Spelling extends Module { forward(a) { return fn(a); } }

  const traced = printModule(await trace((a) => fn(a), [x], { name: 'traced' }));

  let ir = null;
  const compiled = compile(new Spelling(), [x], {
    target: CPUTarget(),
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterGraphPasses: true },
      sink: (e) => { if (e.type === 'ir_snapshot') ir = e.text; },
    },
  });
  await compiled._ready;

  const body = ir.split('\n').slice(2, -2).join('\n');
  console.log(`=== ${label} ===`);
  console.log(`  traced:      ${traced.split('\n').length - 4} operations`);
  console.log(`  canonical:\n${body}`);
  canonical.set(label, body);
}

const forms = new Set(canonical.values());
console.log(`\n${canonical.size} spellings collapsed to ${forms.size} canonical form(s).`);
