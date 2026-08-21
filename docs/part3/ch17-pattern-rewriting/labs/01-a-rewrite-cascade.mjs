import {
  tensor, Module, compile, trace, printModule, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[1, 2], [3, 4]], { dtype: 'i32' });

class LongWayRound extends Module {
  forward(a) { return a.transpose(1, 0).transpose(1, 0).add(0).mul(1); }
}

console.log('=== traced ===');
console.log(printModule(await trace((a) => new LongWayRound().forward(a), [x])));

let ir = null;
let pending = null;
const rows = [];

const compiled = compile(new LongWayRound(), [x], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.DEBUG,
    irSnapshot: { afterGraphPasses: true },
    sink: (e) => {
      if (e.type === 'pass_detail') pending = e;
      if (e.type === 'pass') { rows.push({ ...e, detail: pending }); pending = null; }
      if (e.type === 'ir_snapshot') ir = e.text;
    },
  },
});
await compiled._ready;

console.log('=== what the pattern applicator did ===');
let round = 0;
for (const e of rows) {
  if (e.passName === 'canonicalize') round++;
  if (!e.changed) continue;
  const d = e.detail;
  const detail = !d ? ''
    : d.passName === 'PatternApplicator' ? `${d.totalRewrites} rewrite(s) from a set of ${d.patternCount} patterns`
    : `${d.passName} reports ${Object.entries(d).filter(([k]) => !['type', 'passName', 'level', 'timestamp'].includes(k)).map(([k, v]) => `${k}=${v}`).join(', ')}`;
  console.log(`  round ${round}  ${e.passName.padEnd(20)} ${e.opCountBefore} -> ${e.opCountAfter} ops   ${detail}`);
}

console.log('\n=== after graph passes ===');
console.log(ir);
