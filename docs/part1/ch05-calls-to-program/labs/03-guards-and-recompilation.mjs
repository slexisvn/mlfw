import {
  tensor, randn, Module, Linear, compile, CPUTarget, TraceLevel, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

class Net extends Module {
  constructor() { super(); this.l = new Linear(3, 2); }
  forward(t) { return this.l.forward(t).relu(); }
}

let compilations = 0;
const model = new Net();
const x4 = randn([4, 3]);

const compiled = compile(model, [x4], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.INFO,
    sink: (e) => { if (e.type === 'phase' && e.phase === 'compile' && e.action === 'start') compilations++; },
  },
});
await compiled._ready;
console.log(`after building with a [4,3] input          : ${compilations} compilation(s)`);

await compiled(randn([4, 3]));
console.log(`after another [4,3] input                  : ${compilations}`);

await compiled(randn([8, 3]));
console.log(`after an [8,3] input                       : ${compilations}`);

await compiled(randn([4, 3]));
console.log(`after going back to [4,3]                  : ${compilations}`);

await compiled(randn([16, 3]));
await compiled(randn([32, 3]));
console.log(`after [16,3] and [32,3]                    : ${compilations}`);

const dyn = compile(model, [x4], { target: CPUTarget(), dynamic_shapes: [true] });
await dyn._ready;
const outs = [];
for (const n of [4, 8, 16, 32]) outs.push((await dyn(randn([n, 3]))).shape.join('x'));
console.log(`\nwith dynamic_shapes: one kernel served     : ${outs.join(', ')}`);
