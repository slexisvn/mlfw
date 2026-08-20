import {
  tensor, Module, compile, CPUTarget, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const x = tensor([[NaN, Infinity, -Infinity, -0.0]]);
const show = (a) => '[' + a.map(v => Object.is(v, -0) ? '-0' : String(v)).join(', ') + ']';

class AddZero extends Module { forward(a) { return a.add(0); } }
class SubZero extends Module { forward(a) { return a.sub(0); } }
class MulOne extends Module { forward(a) { return a.mul(1); } }
class MulZero extends Module { forward(a) { return a.mul(0); } }
class SubSelf extends Module { forward(a) { return a.sub(a); } }
class DivSelf extends Module { forward(a) { return a.div(a); } }

const CASES = [
  ['x + 0', AddZero], ['x - 0', SubZero], ['x * 1', MulOne],
  ['x * 0', MulZero], ['x - x', SubSelf], ['x / x', DivSelf],
];

async function run(label, fastMath) {
  console.log(`=== ${label} ===`);
  console.log(`input${' '.repeat(4)} ${show(x.toArray()[0])}`);
  for (const [name, Klass] of CASES) {
    const eager = new Klass().forward(x).toArray()[0];
    const compiled = compile(new Klass(), [x], {
      target: CPUTarget(),
      optimization: { fastMath },
    });
    await compiled._ready;
    const out = (await compiled(x)).toArray()[0];
    const agree = eager.every((v, i) => Object.is(v, out[i]));
    console.log(`${name.padEnd(8)} eager ${show(eager).padEnd(30)} compiled ${show(out).padEnd(30)} ${agree ? '' : '<-- DIFFERENT'}`);
  }
  console.log();
}

await run('default: fastMath off', false);
await run('fastMath on', true);
