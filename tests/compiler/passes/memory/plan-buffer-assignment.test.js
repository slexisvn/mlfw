import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../../src/compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { splitGraphForNative } from '../../../../src/compiler/passes/partition/cublas_split.js';
import { assignPlanBuffers, computePlanDonations, planMemoryReport } from '../../../../src/compiler/passes/memory/plan_buffer_assignment.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { FusionPass } from '../../../../src/compiler/passes/fusion/fusion_pass.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';

const F32 = ScalarType.F32;
const t2 = (r, c) => new TensorType([r, c], F32);

function mlp(name, N) {
  return buildFunction(name, [t2(N, N), t2(N, N), t2(N, N), t2(N, N)], [t2(N, N)], (b, [x, w1, w2, w3]) => {
    const h1 = b.relu(b.matmul(x, w1).getResult(0)).getResult(0);
    const h2 = b.relu(b.matmul(h1, w2).getResult(0)).getResult(0);
    b.returnOp([b.matmul(h2, w3).getResult(0)]);
  });
}

function splitPlan(func) {
  const mod = new GraphModule(func.name + '_mod');
  mod.addFunction(func);
  const split = splitGraphForNative(mod, 1);
  return split ? { plan: split.plan, mod } : null;
}

// Replays the plan against the assignment: a step may only read a slot whose
// buffer still holds that slot's data. Independent of how the assignment was built.
function firstUnsafeRead(plan) {
  const bufOf = (s) => (plan.buffers ? plan.buffers.slotBuffer[s] : s);
  const written = new Set();
  for (const step of plan.steps) for (const s of step.outputSlots) written.add(s);
  const owner = new Map();
  for (let s = 0; s < plan.numSlots; s++) {
    if (!written.has(s)) owner.set(bufOf(s), s);
  }
  for (let k = 0; k < plan.steps.length; k++) {
    for (const s of plan.steps[k].inputSlots) {
      if (owner.get(bufOf(s)) !== s) return { step: k, slot: s, holder: owner.get(bufOf(s)) };
    }
    for (const s of plan.steps[k].outputSlots) owner.set(bufOf(s), s);
  }
  for (const fx of plan.returnFixups || []) {
    if (fx.kind === 'copy' && owner.get(bufOf(fx.srcSlot)) !== fx.srcSlot) return { step: 'fixup', slot: fx.srcSlot };
  }
  return null;
}

describe('module-level plan buffer assignment', () => {
  it('reuses slots with disjoint lifetimes across the plan steps', () => {
    const { plan } = splitPlan(mlp('mlp_reuse', 16));
    expect(plan.steps.length, 'graph split into several steps').toBeGreaterThan(2);

    const buffers = assignPlanBuffers(plan, []);
    expect(buffers, 'assignment produced').toBeTruthy();
    expect(buffers.slotBuffer.length).toBe(plan.numSlots);
    expect(buffers.bufferBytes.length, 'fewer buffers than slots').toBeLessThan(plan.numSlots);

    plan.buffers = buffers;
    expect(firstUnsafeRead(plan), 'every read still sees its own data').toBeNull();

    const distinct = new Set(buffers.slotBuffer).size;
    expect(distinct).toBe(buffers.bufferBytes.length);
  });

  it('keeps arguments, plan returns and fixup sources on private buffers', () => {
    const { plan } = splitPlan(mlp('mlp_pins', 16));
    const buffers = assignPlanBuffers(plan, []);
    const owners = new Map();
    for (let s = 0; s < plan.numSlots; s++) {
      const b = buffers.slotBuffer[s];
      owners.set(b, (owners.get(b) || 0) + 1);
    }
    for (const s of plan.argSlots) {
      expect(owners.get(buffers.slotBuffer[s]), `arg slot ${s} is private`).toBe(1);
    }
    for (const fx of plan.returnFixups || []) {
      if (fx.kind === 'copy') expect(owners.get(buffers.slotBuffer[fx.srcSlot]), 'fixup source is private').toBe(1);
    }
  });

  it('shrinks the reported footprint below the sum of per-slot allocations', () => {
    const { plan } = splitPlan(mlp('mlp_bytes', 32));
    const buffers = assignPlanBuffers(plan, []);
    const report = planMemoryReport(plan, buffers);
    expect(report.slotBytes, 'plan has intermediates').toBeGreaterThan(0);
    expect(report.bufferBytes).toBeLessThan(report.slotBytes);
  });

  it('donates a dying elementwise input to its output and stays replayable', () => {
    const { plan, mod } = splitPlan(mlp('mlp_donate', 16));
    const donations = computePlanDonations(mod, plan);
    expect(donations.length, 'relu steps donate their input').toBeGreaterThan(0);

    const buffers = assignPlanBuffers(plan, donations);
    expect(buffers.donated).toBeGreaterThan(0);
    const applied = donations.filter(d => buffers.slotBuffer[d.from] === buffers.slotBuffer[d.to]);
    expect(applied.length).toBe(buffers.donated);

    plan.buffers = buffers;
    expect(firstUnsafeRead(plan), 'donation keeps every read valid').toBeNull();
  });

  it('sees through a fused elementwise region', () => {
    const func = buildFunction('fused_ew', [t2(16, 16), t2(16, 16), t2(16, 16), t2(16, 16)], [t2(16, 16)], (b, [x, w1, bias, w2]) => {
      const h = b.matmul(x, w1).getResult(0);
      const act = b.tanh(b.add(h, bias).getResult(0)).getResult(0);
      b.returnOp([b.matmul(act, w2).getResult(0)]);
    });
    expect(new FusionPass({}).run(func)).toBe(PassResult.CHANGED);

    const { plan, mod } = splitPlan(func);
    const fusionStep = plan.steps.findIndex(s => [...mod.getFunction(s.name).ops()].some(op => op.opName === 'fusion'));
    expect(fusionStep, 'the elementwise partition is a fusion region').toBeGreaterThanOrEqual(0);

    const donations = computePlanDonations(mod, plan);
    expect(donations.map(d => d.step)).toContain(fusionStep);

    plan.buffers = assignPlanBuffers(plan, donations);
    expect(plan.buffers.donated).toBeGreaterThan(0);
    expect(firstUnsafeRead(plan)).toBeNull();
  });

  it('refuses a same-shape step that reindexes its input', () => {
    const func = buildFunction('reindex', [t2(16, 16), t2(16, 16), t2(16, 16)], [t2(16, 16)], (b, [x, w1, w2]) => {
      const h = b.matmul(x, w1).getResult(0);
      const moved = b.tanh(b.transpose(h, [1, 0]).getResult(0)).getResult(0);
      b.returnOp([b.matmul(moved, w2).getResult(0)]);
    });
    const { plan, mod } = splitPlan(func);
    const transposeStep = plan.steps.findIndex(s => [...mod.getFunction(s.name).ops()].some(op => op.opName === 'transpose'));
    expect(transposeStep).toBeGreaterThanOrEqual(0);
    expect(computePlanDonations(mod, plan).map(d => d.step)).not.toContain(transposeStep);
  });

  it('never donates through a step that is not index-preserving', () => {
    const { plan, mod } = splitPlan(mlp('mlp_no_donate', 16));
    const donations = computePlanDonations(mod, plan);
    for (const d of donations) {
      const func = mod.getFunction(plan.steps[d.step].name);
      const names = [...func.ops()].map(op => op.opName);
      expect(names, `step ${plan.steps[d.step].name} has no matmul`).not.toContain('dot');
    }
  });

  // Both branches die inside the elementwise step, so liveness alone has to open a
  // third buffer for its result; donation folds it onto a branch that dies there.
  it('donation removes the buffer liveness alone cannot free', () => {
    const branchy = (name) => buildFunction(name, [t2(24, 24), t2(24, 24), t2(24, 24), t2(24, 24)], [t2(24, 24)], (b, [x, w1, w2, w3]) => {
      const t1 = b.matmul(x, w1).getResult(0);
      const t2v = b.matmul(x, w2).getResult(0);
      const sum = b.add(b.relu(t1).getResult(0), b.relu(t2v).getResult(0)).getResult(0);
      b.returnOp([b.matmul(sum, w3).getResult(0)]);
    });

    const plain = splitPlan(branchy('branch_a'));
    const base = assignPlanBuffers(plain.plan, []);

    const donatable = splitPlan(branchy('branch_b'));
    const donations = computePlanDonations(donatable.mod, donatable.plan);
    const donated = assignPlanBuffers(donatable.plan, donations);

    expect(donated.donated, 'both relu partitions donate').toBe(2);
    expect(donated.bufferBytes.length).toBe(base.bufferBytes.length - 1);

    donatable.plan.buffers = donated;
    expect(firstUnsafeRead(donatable.plan)).toBeNull();
  });
});

describe('plan buffer assignment through the compiler', () => {
  const splitTarget = () => CPUTarget({ attrs: { graphSplit: { matmul: 1 } } });

  function inputs(N) {
    const gen = (s) => Float32Array.from({ length: N * N }, (_, i) => Math.sin(i * 0.03 + s) * 0.5);
    return [gen(1), gen(2), gen(3), gen(4)];
  }

  async function runPlanned(target, opts, N) {
    const result = compileGraph(mlp('mlp_exec', N), target, opts);
    const plan = result.module.executionPlan;
    const out = new Float32Array(N * N);
    await result.module.runPlanAsync(plan, [...inputs(N), out]);
    return { out, plan };
  }

  it('attaches an assignment to the compiled plan and matches the unassigned run bit-for-bit', async () => {
    const N = 16;
    const reference = new Float32Array(N * N);
    compileGraph(mlp('mlp_exec', N), CPUTarget()).run('mlp_exec', ...inputs(N), reference);

    const off = await runPlanned(splitTarget(), { memory: { planReuse: false } }, N);
    expect(off.plan, 'target thresholds split the graph').toBeTruthy();
    expect(off.plan.buffers, 'assignment disabled').toBeUndefined();

    const on = await runPlanned(splitTarget(), {}, N);
    expect(on.plan.buffers, 'assignment attached').toBeTruthy();
    expect(on.plan.buffers.bufferBytes.length).toBeLessThan(on.plan.numSlots);

    expect([...off.out]).toEqual([...reference]);
    expect([...on.out]).toEqual([...reference]);
  });

  it('runs correctly with donation disabled', async () => {
    const N = 16;
    const reference = new Float32Array(N * N);
    compileGraph(mlp('mlp_exec', N), CPUTarget()).run('mlp_exec', ...inputs(N), reference);

    const noDonation = await runPlanned(splitTarget(), { memory: { planDonation: false } }, N);
    expect(noDonation.plan.buffers.donated).toBe(0);
    expect([...noDonation.out]).toEqual([...reference]);

    const donation = await runPlanned(splitTarget(), {}, N);
    expect(donation.plan.buffers.donated).toBeGreaterThan(0);
    expect([...donation.out]).toEqual([...reference]);
  });
});
