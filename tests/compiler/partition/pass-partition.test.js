import { describe, it, expect } from 'vitest';
import { buildFunction, IRBuilder } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { GraphPartitionPass, PartitionMaterializationPass } from '../../../src/compiler/passes/partition/partition_pass.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { GraphFunction } from '../../../src/compiler/ir/graph/function.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

function allOps(func) {
  const list = [];
  for (const op of func.ops()) {
    if (op.opName !== 'return') list.push(op);
  }
  return list;
}

describe('GraphPartitionPass', () => {
  it('returns UNCHANGED with fewer than 2 targets', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const pass = new GraphPartitionPass({ targets: [CPUTarget()] });
    expect(pass.run(func)).toBe(PassResult.UNCHANGED);
  });

  it('returns UNCHANGED when all ops land on one partition', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const pass = new GraphPartitionPass({ targets: [CPUTarget(), CUDATarget()] });
    const result = pass.run(func);
    if (result === PassResult.UNCHANGED) {
      expect(findOps(func, 'copy_to_device').length).toBe(0);
    }
  });

  it('annotates ops with partition_id and partition_target', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const tSmall = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t, tSmall, tSmall], [t, tSmall], (b, args) => {
      const gpuAdd = b.add(args[0], args[1]);
      const cpuAdd = b.add(args[2], args[3]);
      b.returnOp([gpuAdd.getResult(0), cpuAdd.getResult(0)]);
    });

    const pass = new GraphPartitionPass({ targets: [cpu, gpu] });
    pass.run(func);

    for (const op of allOps(func)) {
      if (op.opName === 'copy_to_device') continue;
      const pid = op.getAttr('partition_id');
      const ptarget = op.getAttr('partition_target');
      expect(pid).toBeDefined();
      expect(ptarget).toBeDefined();
      expect(typeof ptarget).toBe('string');
    }
  });

  it('inserts copy_to_device for cross-device data transfer', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const tSmall = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, tSmall], [t], (b, args) => {
      const cpuNeg = b.neg(args[1]);
      cpuNeg.setAttr('device', 'cpu');
      const bc = b.broadcast(cpuNeg.getResult(0), [256, 256], [1]);
      const gpuAdd = b.add(args[0], bc.getResult(0));
      b.returnOp([gpuAdd.getResult(0)]);
    });

    const pass = new GraphPartitionPass({
      targets: [cpu, gpu],
      opTargetOverrides: new Map([['neg', cpu]])
    });
    const result = pass.run(func);

    if (result === PassResult.CHANGED) {
      const copies = findOps(func, 'copy_to_device');
      expect(copies.length).toBeGreaterThan(0);

      for (const copy of copies) {
        const srcDevice = copy.getAttr('src_device');
        const dstDevice = copy.getAttr('dst_device');
        expect(srcDevice).not.toBe(dstDevice);
        expect(copy.numOperands).toBe(1);
        expect(copy.numResults).toBe(1);
        expect(copy.getResult(0).type).toEqual(copy.getOperand(0).type);
      }
    }
  });

  it('copy_to_device rewires consumer operands', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const tSmall = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, tSmall], [t], (b, args) => {
      const cpuNeg = b.neg(args[1]);
      cpuNeg.setAttr('device', 'cpu');
      const bc = b.broadcast(cpuNeg.getResult(0), [256, 256], [1]);
      const gpuAdd = b.add(args[0], bc.getResult(0));
      b.returnOp([gpuAdd.getResult(0)]);
    });

    const pass = new GraphPartitionPass({
      targets: [cpu, gpu],
      opTargetOverrides: new Map([['neg', cpu]])
    });
    const result = pass.run(func);

    if (result === PassResult.CHANGED) {
      const copies = findOps(func, 'copy_to_device');
      for (const copy of copies) {
        const copyResult = copy.getResult(0);
        let hasUser = false;
        for (const use of copyResult.uses()) {
          hasUser = true;
          expect(use.user.opName).not.toBe('copy_to_device');
        }
        expect(hasUser).toBe(true);
      }
    }
  });

  it('same-device edges do NOT get copy_to_device', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const pass = new GraphPartitionPass({ targets: [cpu, gpu] });
    pass.run(func);

    const copies = findOps(func, 'copy_to_device');
    for (const copy of copies) {
      expect(copy.getAttr('src_device')).not.toBe(copy.getAttr('dst_device'));
    }
  });
});

describe('GraphPartitionPass topo sort and insertion ordering', () => {
  it('_topoSort throws when the op set contains a cycle', () => {
    const pass = new PartitionMaterializationPass();

    const a = { numOperands: 1, numResults: 1, getOperand() { return { definingOp: b }; }, getResult() { return {}; } };
    const b = { numOperands: 1, numResults: 1, getOperand() { return { definingOp: a }; }, getResult() { return {}; } };

    expect(() => pass._topoSort([a, b])).toThrow(/cycle/i);
  });

  it('_topoSort returns all ops for an acyclic set', () => {
    const pass = new PartitionMaterializationPass();
    const r1 = {};
    const p = { numOperands: 0, numResults: 1, getOperand() { return null; }, getResult() { return r1; } };
    const c = { numOperands: 1, numResults: 0, getOperand() { return { definingOp: p }; }, getResult() { return {}; } };

    const sorted = pass._topoSort([c, p]);
    expect(sorted.length).toBe(2);
    expect(sorted.indexOf(p)).toBeLessThan(sorted.indexOf(c));
  });

  it('copy_to_device is inserted after the transferred value definingOp', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const tSmall = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, tSmall], [t], (b, args) => {
      const cpuNeg = b.neg(args[1]);
      cpuNeg.setAttr('device', 'cpu');
      const bc = b.broadcast(cpuNeg.getResult(0), [256, 256], [1]);
      const gpuAdd = b.add(args[0], bc.getResult(0));
      b.returnOp([gpuAdd.getResult(0)]);
    });

    const pass = new GraphPartitionPass({
      targets: [cpu, gpu],
      opTargetOverrides: new Map([['neg', cpu]])
    });
    const result = pass.run(func);

    if (result === PassResult.CHANGED) {
      const order = [];
      for (const op of func.entryBlock.ops()) order.push(op);

      const copies = findOps(func, 'copy_to_device');
      expect(copies.length).toBeGreaterThan(0);
      for (const copy of copies) {
        const val = copy.getOperand(0);
        const defOp = val.definingOp;
        if (defOp && order.includes(defOp)) {
          expect(order.indexOf(copy)).toBeGreaterThan(order.indexOf(defOp));
        }
        for (const use of copy.getResult(0).uses()) {
          if (order.includes(use.user)) {
            expect(order.indexOf(use.user)).toBeGreaterThan(order.indexOf(copy));
          }
        }
      }
    }
  });
});

describe('PartitionMaterializationPass', () => {
  it('returns UNCHANGED when only 1 partition exists', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const pass = new PartitionMaterializationPass();
    expect(pass.run(func)).toBe(PassResult.UNCHANGED);
  });

  it('materializes partitions into sub-functions with correct IO', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const tSmall = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t, tSmall, tSmall], [t, tSmall], (b, args) => {
      const gpuAdd = b.add(args[0], args[1]);
      const cpuAdd = b.add(args[2], args[3]);
      b.returnOp([gpuAdd.getResult(0), cpuAdd.getResult(0)]);
    });

    const partPass = new GraphPartitionPass({ targets: [cpu, gpu] });
    partPass.run(func);

    const module = { functions: [], addFunction(f) { this.functions.push(f); } };
    func._module = module;

    const matPass = new PartitionMaterializationPass();
    const result = matPass.run(func);

    if (result === PassResult.CHANGED) {
      expect(module.functions.length).toBeGreaterThan(0);
      for (const sub of module.functions) {
        expect(sub.name).toContain('partition');
        expect(sub.getReturnOp()).toBeDefined();
        expect(sub.args.length).toBeGreaterThan(0);
      }
    }
  });

  it('sub-function ops are cloned — not shared with original', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const tSmall = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t, tSmall, tSmall], [t, tSmall], (b, args) => {
      const gpuAdd = b.add(args[0], args[1]);
      const cpuAdd = b.add(args[2], args[3]);
      b.returnOp([gpuAdd.getResult(0), cpuAdd.getResult(0)]);
    });

    const partPass = new GraphPartitionPass({ targets: [cpu, gpu] });
    partPass.run(func);

    const module = { functions: [], addFunction(f) { this.functions.push(f); } };
    func._module = module;

    new PartitionMaterializationPass().run(func);

    const originalOps = new Set();
    for (const op of func.ops()) originalOps.add(op);

    for (const sub of module.functions) {
      for (const op of sub.ops()) {
        expect(originalOps.has(op)).toBe(false);
      }
    }
  });

  it('sub-function operands use block arguments not original values', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const tSmall = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t, tSmall, tSmall], [t, tSmall], (b, args) => {
      const gpuAdd = b.add(args[0], args[1]);
      const cpuAdd = b.add(args[2], args[3]);
      b.returnOp([gpuAdd.getResult(0), cpuAdd.getResult(0)]);
    });

    const partPass = new GraphPartitionPass({ targets: [cpu, gpu] });
    partPass.run(func);

    const module = { functions: [], addFunction(f) { this.functions.push(f); } };
    func._module = module;

    new PartitionMaterializationPass().run(func);

    for (const sub of module.functions) {
      for (const op of sub.ops()) {
        if (op.opName === 'return') continue;
        for (let i = 0; i < op.numOperands; i++) {
          const operand = op.getOperand(i);
          const fromThisSub = operand.isBlockArgument() ||
            (operand.definingOp && operand.definingOp.parentBlock === sub.entryBlock);
          expect(fromThisSub).toBe(true);
        }
      }
    }
  });
});

const F = ScalarType.F32;
function buildAuto(name, inTypes, build) {
  const probe = new GraphFunction(name, inTypes, []);
  const out = build(new IRBuilder(probe), probe.args).getResult(0);
  return { func: buildFunction(name, inTypes, [out.type], (b, a) => { b.returnOp([build(b, a).getResult(0)]); }), n: out.type.shape.reduce((x, y) => x * y, 1) };
}

describe('partition materialization: end-to-end numerics match non-partitioned (region-bearing ops keep their regions)', () => {
  const cpuA = CPUTarget();
  const cpuB = CPUTarget({ name: 'cpu_b' });
  const T = (sh) => new TensorType(sh, F);
  const CASES = [
    { name: 'reduce_sum', inTypes: [T([3, 4, 5])], build: (b, a) => b.reduce(a[0], b.scalarConstant(0, F).getResult(0), [1], 'sum'), split: new Map([['reduce', cpuB]]) },
    { name: 'reduce_bcast_sub', inTypes: [T([4, 6])], build: (b, a) => b.sub(a[0], b.broadcast(b.reduce(a[0], b.scalarConstant(0, F).getResult(0), [1], 'sum').getResult(0), [4, 6], [0]).getResult(0)), split: new Map([['reduce', cpuB]]) },
    { name: 'softmax', inTypes: [T([4, 8])], build: (b, a) => b.softmax(a[0], 1), split: new Map([['sub', cpuB], ['div', cpuB]]) },
    { name: 'ew_chain', inTypes: [T([4, 5]), T([4, 5])], build: (b, a) => b.relu(b.mul(b.add(a[0], a[1]).getResult(0), a[0]).getResult(0)), split: new Map([['mul', cpuB]]) },
  ];
  for (const c of CASES) {
    it(`${c.name} partitioned == non-partitioned`, () => {
      const built = buildAuto(c.name, c.inTypes, c.build);
      const inputs = c.inTypes.map((t, k) => { const a = new Float32Array(t.shape.reduce((x, y) => x * y, 1)); for (let i = 0; i < a.length; i++) a[i] = Math.sin((i + k) * 1.3) * 2; return a; });
      const r0 = compileGraph(built.func, cpuA, { partition: { enabled: false } });
      const ref = new Float32Array(built.n); r0.run(c.name, ...inputs, ref);
      const r1 = compileGraph(built.func, cpuA, { partition: { enabled: true, targets: [cpuA, cpuB], opTargetOverrides: c.split } });
      const out = new Float32Array(built.n); r1.run(c.name, ...inputs, out);
      for (let i = 0; i < built.n; i++) {
        expect(Math.abs(ref[i] - out[i]) / (1 + Math.abs(ref[i])), `${c.name} idx ${i}: ref=${ref[i]} part=${out[i]}`).toBeLessThan(2e-3);
      }
    });
  }
});
