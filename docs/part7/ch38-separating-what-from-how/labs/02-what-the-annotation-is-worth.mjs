import {
  compile, CPUTarget, WasmTarget, CUDATarget, randn, resetVarCounter,
} from '../../_internals.mjs';

async function source(target, scheduling, shape) {
  resetVarCounter();
  const x = randn(shape);
  const compiled = compile({ forward: (a) => a.mul(2.0) }, [x], {
    target, fusion: { enabled: false }, ...(scheduling ? { scheduling } : {}),
  });
  try {
    await compiled(x);
  } catch (e) {
  }
  return compiled.source().split('\n').slice(1).join('\n').trimEnd();
}

async function emit(label, target, scheduling, shape = [100]) {
  console.log(`=== ${label} ===`);
  console.log(await source(target, scheduling, shape));
  console.log();
}

await emit('CPU, scheduling off (the shipped default)', CPUTarget(), null);
await emit('CPU, scheduling on', CPUTarget(), { enabled: true });

console.log('Neither `@parallel` nor `@vectorized` appears in the emitted JavaScript.');
console.log('The CPU backend reads exactly one loop kind, ForKind.UNROLLED');
console.log('(backend/cpu/codegen.ts:231); the other four are ignored. What the');
console.log('schedule did contribute is the split — two loops and a guard where');
console.log('there was one loop and no guard.\n');

const wat = await source(WasmTarget({ numCores: 4 }), { enabled: true }, [4096]);
const simd = wat.split('\n').filter((l) => /f32x4|v128/.test(l));
const par = wat.split('\n').filter((l) => /_par_start|_par_end/.test(l));

console.log('=== WASM, 4 cores, scheduling on ===');
console.log(`  ${wat.split('\n').length} lines of WAT, of which`);
console.log(`  ${par.length} mention the worker pool's slice of the parallel loop:`);
for (const l of par.slice(0, 5)) console.log(`      ${l.trim()}`);
console.log(`  ${simd.length} are SIMD, e.g.`);
for (const l of simd.slice(2, 6)) console.log(`      ${l.trim()}`);
console.log();
console.log('Same two annotations, a backend that spends them: `_par_start`/`_par_end`');
console.log("are the worker pool's slice of the parallel loop, and the `f32x4.*`");
console.log('opcodes are the vectorised one. On WASM the annotations are the schedule.\n');

await emit('CUDA, DEFAULT scheduling config', CUDATarget(), null, [4096]);
await emit('CUDA, scheduling.enabled = true', CUDATarget(), { enabled: true }, [4096]);

console.log('The default CUDA configuration is `{ gpuTiling: true }`');
console.log('(backend/target.ts:225), and `gpuTiling` alone reaches only');
console.log('`applyDeterministicGpuSchedule`, which recognises a matmul or a');
console.log('convolution and nothing else. `SchedulePass.run` then reads');
console.log('`if (!handled && sCfg.enabled)` (schedule_pass.ts:61) and, with');
console.log('`enabled` false, skips the policy. So an elementwise CUDA kernel');
console.log('compiled with the shipped defaults is a serial loop inside a');
console.log('__global__ function: one thread does all 4096 elements.');
console.log('WebGPU declares `{ enabled: true }` (target.ts:261) and does not');
console.log('have the problem. The difference between the two targets is one');
console.log('key in the attribute table.');
