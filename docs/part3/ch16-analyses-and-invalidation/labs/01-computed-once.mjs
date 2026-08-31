import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
}

let computed = [];
let recording = true;

class OpCount {
  static get name() { return 'op_count'; }
  static get depKey() { return 'opCount'; }
  static get dependencies() { return []; }
  static compute(func) {
    if (recording) computed.push('op_count');
    return func.numOps();
  }
}

class Fanout {
  static get name() { return 'fanout'; }
  static get depKey() { return 'fanout'; }
  static get dependencies() { return [OpCount]; }
  static compute(func, deps) {
    if (recording) computed.push('fanout');
    let operands = 0;
    for (const op of func.ops()) operands += op.numOperands;
    return operands / Math.max(deps.opCount, 1);
  }
}

function truth(func) {
  recording = false;
  const value = Fanout.compute(func, { opCount: OpCount.compute(func) });
  recording = true;
  return value;
}

async function run(label, preserved) {
  const instrumented = new WeakSet();
  const rows = [];
  const compiled = compile(build(), [x], {
    target: CPUTarget(),
    passContext: {
      shouldRun(pass) {
        if (!(pass.preservedAnalyses instanceof Set)) return true;
        if (!instrumented.has(pass)) {
          instrumented.add(pass);
          for (const a of preserved) pass.preservedAnalyses.add(a);
          const inner = pass.run.bind(pass);
          pass.run = (target, analyses) => {
            if (typeof target.numOps !== 'function') return inner(target, analyses);
            computed = [];
            const served = analyses.getAnalysis(Fanout, target);
            const actual = truth(target);
            const how = computed.length === 0 ? 'from cache' : 'recomputed ' + computed.join('+');
            const result = inner(target, analyses);
            rows.push(`  ${pass.name.padEnd(21)} v${String(target.version).padEnd(4)} ` +
              `${how.padEnd(26)} fanout=${served.toFixed(2)} ` +
              `${served === actual ? '' : `<- WRONG, the IR says ${actual.toFixed(2)}`}`);
            return result;
          };
        }
        return true;
      },
    },
  });
  await compiled._ready;
  console.log(`=== ${label} ===`);
  for (const r of rows) console.log(r.trimEnd());
  const recomputes = rows.filter(r => r.includes('recomputed')).length;
  const wrong = rows.filter(r => r.includes('WRONG')).length;
  console.log(`  ${recomputes} of ${rows.length} runs recomputed; ${wrong} were served a stale answer\n`);
}

await run('no pass preserves anything (the default)', []);
await run('every pass preserves both analyses', [OpCount, Fanout]);
