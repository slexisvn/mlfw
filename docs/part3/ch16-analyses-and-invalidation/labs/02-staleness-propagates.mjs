import {
  tensor, Linear, ReLU, Sequential, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

function build() {
  manual_seed(0);
  return new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
}

let computed = [];

class OpCount {
  static get name() { return 'op_count'; }
  static get depKey() { return 'opCount'; }
  static get dependencies() { return []; }
  static compute(func) { computed.push('op_count'); return func.numOps(); }
}

class Fanout {
  static get name() { return 'fanout'; }
  static get depKey() { return 'fanout'; }
  static get dependencies() { return [OpCount]; }
  static compute(func, deps) {
    computed.push('fanout');
    let operands = 0;
    for (const op of func.ops()) operands += op.numOperands;
    return operands / Math.max(deps.opCount, 1);
  }
}

class Shape {
  static get name() { return 'shape'; }
  static get depKey() { return 'shape'; }
  static get dependencies() { return [Fanout]; }
  static compute(func, deps) { computed.push('shape'); return deps.fanout > 1.5 ? 'wide' : 'narrow'; }
}

async function run(label, preserved) {
  const instrumented = new WeakSet();
  const tally = new Map([['op_count', 0], ['fanout', 0], ['shape', 0]]);
  let runs = 0;
  const compiled = compile(build(), [x], {
    target: CPUTarget(),
    passContext: {
      shouldRun(pass) {
        if (!instrumented.has(pass)) {
          instrumented.add(pass);
          for (const a of preserved) pass.preservedAnalyses.add(a);
          const inner = pass.run.bind(pass);
          pass.run = (target, analyses) => {
            if (typeof target.numOps !== 'function') return inner(target, analyses);
            computed = [];
            analyses.getAnalysis(Shape, target);
            runs++;
            for (const name of computed) tally.set(name, tally.get(name) + 1);
            return inner(target, analyses);
          };
        }
        return true;
      },
    },
  });
  await compiled._ready;
  const names = preserved.map(a => a.name);
  console.log(`${label.padEnd(38)} preserves {${names.join(', ') || 'nothing'}}`);
  console.log(`  over ${runs} pass runs: ` +
    [...tally].map(([n, c]) => `${n} recomputed ${c}x`).join(', ') + '\n');
}

console.log('dependency chain: shape -> fanout -> op_count\n');
await run('all three declared preserved', [OpCount, Fanout, Shape]);
await run('the two dependents, not the root', [Fanout, Shape]);
await run('the root only', [OpCount]);
await run('nothing', []);
