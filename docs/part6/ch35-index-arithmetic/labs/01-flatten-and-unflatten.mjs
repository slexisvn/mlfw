import {
  compile, CPUTarget, TraceLevel, randn,
} from '../../../../dist/index.node.js';

// A reshape is the purest index problem there is: no arithmetic on the data,
// only arithmetic on the coordinates. The rule always emits the same thing —
// flatten the output coordinate, then divide and modulo it back apart. What
// differs is how much of that survives the simplifier.

function normalise(text) {
  const names = new Map();
  return text
    .replace(/buf_(\d+)/g, (_, n) => {
      if (!names.has(n)) names.set(n, `b${names.size}`);
      return names.get(n);
    })
    .replace(/v(\d+)_\d+/g, 'i$1')
    .replace(/i(\d+)_\d+/g, 'i$1');
}

async function reshape(inShape, outShape) {
  const snaps = [];
  const x = randn(inShape);
  const compiled = compile({ forward: (t) => t.reshape(outShape) }, [x], {
    target: CPUTarget(),
    fusion: { enabled: false },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterLowering: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  await compiled(x);
  const tir = normalise(snaps[0]).split('\n').find((l) => /^\s+b\d+\[[^\]]*\] = /.test(l)).trim();
  const js = compiled.source().split('\n').find((l) => l.includes('] = ')).trim().replace(/;$/, '');
  const rhs = js.slice(js.indexOf('] = ') + 4);
  return {
    label: `${JSON.stringify(inShape)} -> ${JSON.stringify(outShape)}`,
    tir: tir.slice(tir.indexOf('] = ') + 4),
    js: rhs,
    divs: (rhs.match(/\/|%/g) || []).length,
  };
}

const cases = [
  [[1, 6], [6]],
  [[4, 3], [2, 2, 3]],
  [[2, 2, 3], [4, 3]],
  [[4, 3], [12]],
  [[4, 3], [2, 6]],
  [[4, 3], [3, 4]],
];

console.log('=== what the rule emits, and what survives ===\n');
for (const [inS, outS] of cases) {
  const r = await reshape(inS, outS);
  console.log(`  ${r.label}`);
  console.log(`    TIR : ${r.tir}`);
  console.log(`    JS  : ${r.js}${r.divs === 0 ? '     <- no division left' : ''}`);
}

console.log('\n  Row 1: the divisor covers the whole range of the dividend, so the');
console.log('  quotient is provably 0 and the remainder is provably the dividend.');
console.log('  Row 2: every coefficient in the flat index is a multiple of the');
console.log('  divisor except the last, and that last one fits in one residue class —');
console.log('  the mixed-radix split applies and both operations disappear.');
console.log('  Row 3: the same split removes the outer // 3 and % 3 and leaves the');
console.log('  inner / 2 and % 2, which no split covers.');
console.log('  Rows 4-6: the split does not apply at all, so a division and a modulo');
console.log('  per element reach the backend.');

console.log('\n\n=== the coordinate transforms that need no division at all ===\n');

async function shapeOp(label, fn, shape) {
  const snaps = [];
  const x = randn(shape);
  const compiled = compile({ forward: fn }, [x], {
    target: CPUTarget(),
    fusion: { enabled: false },
    trace: {
      level: TraceLevel.DEBUG,
      irSnapshot: { afterLowering: true },
      sink: (e) => { if (e.type === 'ir_snapshot') snaps.push(e.text); },
    },
  });
  await compiled(x);
  const tir = normalise(snaps[0]).split('\n').find((l) => /^\s+b\d+\[[^\]]*\] = /.test(l)).trim();
  console.log(`  ${label.padEnd(24)} ${tir}`);
}

await shapeOp('transpose(1,0)', (t) => t.transpose(1, 0), [4, 3]);
await shapeOp('slice rows 1..3', (t) => t.narrow(0, 1, 2), [4, 3]);
await shapeOp('x + x            ', (t) => t.add(t), [4, 3]);

console.log('\n  A permutation is a relabelling of the loop variables; a slice is one');
console.log('  addition per axis. Only a reshape has to cross a stride boundary, and');
console.log('  only a reshape needs division.');
