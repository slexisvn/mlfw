import {
  tensor, compileWithBackward, CPUTarget, TraceLevel, ones,
} from '../../../../dist/index.node.js';

// `add` has the simplest VJP rule in the compiler: return the incoming
// gradient, twice. But `x + b` with a broadcast b does not have a gradient
// of b's shape -- somebody has to put it back. This lab finds who.

const x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);   // [2, 3]
const b = tensor([0.5, 0.5, 0.5]);                      // [3], broadcast over rows

async function settle(v) { return v && v.then ? await v : v; }

const snaps = [];
const cf = compileWithBackward({ forward: (a, bias) => a.add(bias).sum() }, [x, b], {
  target: CPUTarget(),
  fusion: { enabled: false },
  trace: {
    level: TraceLevel.DEBUG,
    irSnapshot: { afterGraphPasses: true },
    sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
  },
});

const out = await settle(cf(x, b));
const grads = await settle(cf.backward(ones(out.shape)));

console.log('=== forward: x is [2, 3], b is [3] ===');
console.log(snaps[0].split('\n').filter(l => /func |%\d+ = |return\(/.test(l)).map(l => '  ' + l.trim()).join('\n'));

console.log('\n=== backward ===');
console.log(snaps[1].split('\n').filter(l => /func |%\d+ = |return\(/.test(l)).map(l => '  ' + l.trim()).join('\n'));

console.log('\n=== the two gradients ===');
console.log('  d/dx shape ' + JSON.stringify(grads[0].shape) + '  = ' + JSON.stringify(grads[0].toArray()));
console.log('  d/db shape ' + JSON.stringify(grads[1].shape) + '  = ' + JSON.stringify(grads[1].toArray()));
console.log('\n  the add rule returned the same [2, 3] gradient for both operands;');
console.log('  d/db is [3] and each entry is 2 -- one per row that b was broadcast into.');
