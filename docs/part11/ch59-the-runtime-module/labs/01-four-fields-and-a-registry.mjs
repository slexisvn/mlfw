import {
  RuntimeModule, RuntimeTensor, BackendPipeline, lowerGraphToPrimFunc, firstFunction,
  CPUTarget, WasmTarget, CUDATarget, WebGPUTarget, hasBackend, getBackend, registerBackend,
  trace, randn, manual_seed,
} from '../../_internals.mjs';

manual_seed(2);

const graph = await trace((t) => t.mul(2).add(1).relu(), [randn([4, 4])]);
const func = firstFunction(graph);
const kernelFor = (target) => new BackendPipeline(target).compile(lowerGraphToPrimFunc(func, target));

console.log('=== a compiled kernel has four fields, and the runtime reads all four ===\n');
const targets = [CPUTarget(), WasmTarget(), CUDATarget(), WebGPUTarget()];
console.log(`  ${'target'.padEnd(16)} ${'kind'.padEnd(8)} ${'source'.padStart(7)} ${'async'.padEnd(6)} metadata`);
for (const target of targets) {
  const kernel = kernelFor(target);
  const module = new RuntimeModule('probe');
  module.addCompiledKernel(kernel);
  const keys = Object.keys(kernel.metadata).filter((k) => k !== 'kind');
  console.log(
    `  ${target.name.padEnd(16)} ${kernel.metadata.kind.padEnd(8)} ${String(kernel.source.length).padStart(7)}`
    + ` ${String(module.isAsync(kernel.name)).padEnd(6)} ${keys.join(', ')}`,
  );
}

console.log('\n=== dispatch is one map lookup on metadata.kind ===\n');
for (const kind of ['js', 'wasm', 'cuda', 'webgpu', 'opencl']) {
  console.log(`  getBackend(${JSON.stringify(kind).padEnd(9)}) -> ${hasBackend(kind) ? 'a backend' : 'null'}`);
}

console.log('\n=== loading and calling one, by hand ===\n');
const cpu = kernelFor(CPUTarget());
const module = new RuntimeModule('by-hand');
module.addCompiledKernel(cpu);
console.log('  kernels          ', module.listKernels());
console.log('  instance type    ', typeof module.instantiate(cpu.name));

const input = RuntimeTensor.fromArray([...Array(16).keys()].map((i) => i - 3), [4, 4], 'f32');
const output = RuntimeTensor.zeros([4, 4], 'f32');
module.run(cpu.name, input, output);
console.log('  in  [0..6]       ', [...input.data.slice(0, 6)].join(' '));
console.log('  out [0..6]       ', [...output.data.slice(0, 6)].join(' '));
console.log('  strides of a 4x4 ', output.strides.join(','), ' numel', output.numel, ' rank', output.rank);

console.log('\n=== the second call does not compile again ===\n');
const timed = (fn) => { const t0 = performance.now(); fn(); return performance.now() - t0; };
const fresh = new RuntimeModule('cold');
fresh.addCompiledKernel(cpu);
const cold = timed(() => fresh.run(cpu.name, input, output));
const warm = timed(() => fresh.run(cpu.name, input, output));
console.log(`  first run ${cold.toFixed(3)}ms   second run ${warm.toFixed(3)}ms`);

console.log('\n=== serialize, deserialize, run again ===\n');
const wire = module.serialize();
console.log('  serialized       ', JSON.stringify({ name: wire.name, kernels: wire.kernels.map((k) => ({ name: k.name, target: k.target, kind: k.metadata.kind })) }));
const revived = RuntimeModule.deserialize(wire);
const output2 = RuntimeTensor.zeros([4, 4], 'f32');
revived.run(cpu.name, input, output2);
const identical = [...output.data].every((v, i) => v === output2.data[i]);
console.log('  same numbers     ', identical);

console.log('\n=== a backend is a four-method object, and you can add one ===\n');
registerBackend('counting-js', {
  instantiate(kernel) { return { fn: new Function('return ' + kernel.source)(), calls: 0 }; },
  runSync(inst, tensorArgs, shapeValues) {
    inst.calls++;
    return inst.fn(...(shapeValues ? [...tensorArgs, ...shapeValues] : tensorArgs));
  },
  runAsync(inst, tensorArgs, shapeValues) { return this.runSync(inst, tensorArgs, shapeValues); },
  isAsync() { return false; },
});

const counted = new RuntimeModule('counted');
counted.addCompiledKernel({ ...cpu, name: cpu.name, metadata: { ...cpu.metadata, kind: 'counting-js' } });
const output3 = RuntimeTensor.zeros([4, 4], 'f32');
for (let i = 0; i < 3; i++) counted.run(cpu.name, input, output3);
console.log('  launches counted ', counted.instantiate(cpu.name).calls);
console.log('  same numbers     ', [...output3.data].every((v, i) => v === output.data[i]));
console.log('\n  nothing in src/ changed: a new runtime backend is one registerBackend call.');
