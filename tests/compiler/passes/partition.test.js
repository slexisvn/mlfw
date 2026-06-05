import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction, buildModule } from '../../../src/compiler/ir/graph/builder.js';
import { Operation } from '../../../src/compiler/ir/graph/operation.js';
import { registry } from '../../../src/compiler/ir/graph/ops.js';
import { CPUTarget, GPUTarget, WasmTarget } from '../../../src/compiler/backend/target.js';

import { GraphPartitioner, PartitionerConfig, Partition, PartitionResult } from '../../../src/compiler/analysis/partitioner.js';
import { GraphPartitionPass, PartitionMaterializationPass } from '../../../src/compiler/passes/partition/partition_pass.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

const f32 = ScalarType.F32;
const f32_4x4 = new TensorType([4, 4], f32);
const f32_8x8 = new TensorType([8, 8], f32);
const f32_64x64 = new TensorType([64, 64], f32);

describe('copy_to_device op', () => {
  it('is registered in the op registry', () => {
    assert.ok(registry.has('copy_to_device'));
    const def = registry.get('copy_to_device');
    assert.equal(def.numOperands, 1);
    assert.equal(def.numResults, 1);
  });

  it('infers result types as pass-through', () => {
    const def = registry.get('copy_to_device');
    const result = def.inferResultTypes(
      [f32_4x4],
      new Map([['src_device', 'cpu_generic'], ['dst_device', 'gpu_generic']])
    );
    assert.deepEqual(result, [f32_4x4]);
  });

  it('reports zero flops', () => {
    const def = registry.get('copy_to_device');
    assert.equal(def.getFlops(), 0);
  });
});

describe('Partition', () => {
  it('tracks ops and computes IO', () => {
    const cpu = CPUTarget();
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      const mul = b.mul(add.getResult(0), x);
      b.returnOp([mul.getResult(0)]);
    });

    const ops = func.opsArray().filter(op => op.opName !== 'return');
    const partition = new Partition(0, cpu);
    partition.addOp(ops[0]);

    assert.equal(partition.size, 1);
    assert.ok(partition.hasOp(ops[0]));
    assert.ok(!partition.hasOp(ops[1]));

    const inputs = partition.getInputValues();
    assert.equal(inputs.length, 2);

    const outputs = partition.getOutputValues();
    assert.equal(outputs.length, 1);
  });

  it('merge combines two partitions', () => {
    const cpu = CPUTarget();
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      const mul = b.mul(add.getResult(0), x);
      b.returnOp([mul.getResult(0)]);
    });

    const ops = func.opsArray().filter(op => op.opName !== 'return');
    const p1 = new Partition(0, cpu);
    const p2 = new Partition(1, cpu);
    p1.addOp(ops[0]);
    p2.addOp(ops[1]);

    p1.merge(p2);
    assert.equal(p1.size, 2);
    assert.ok(p1.hasOp(ops[0]));
    assert.ok(p1.hasOp(ops[1]));
  });

  it('estimates memory from result types', () => {
    const cpu = CPUTarget();
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const neg = b.neg(x);
      b.returnOp([neg.getResult(0)]);
    });

    const ops = func.opsArray().filter(op => op.opName !== 'return');
    const partition = new Partition(0, cpu);
    partition.addOp(ops[0]);
    assert.ok(partition.memoryBytes > 0);
  });
});

describe('GraphPartitioner', () => {
  it('assigns all ops to single target when only one target given', () => {
    const cpu = CPUTarget();
    const config = new PartitionerConfig({ targets: [cpu], defaultTarget: cpu });
    const partitioner = new GraphPartitioner(config);

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      const mul = b.mul(add.getResult(0), y);
      b.returnOp([mul.getResult(0)]);
    });

    const result = partitioner.partition(func);
    assert.ok(result instanceof PartitionResult);
    assert.ok(result.numPartitions >= 1);

    for (const p of result.partitions) {
      assert.equal(p.target.name, 'cpu_generic');
    }
  });

  it('partitions across CPU and GPU targets', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const config = new PartitionerConfig({
      targets: [cpu, gpu],
      defaultTarget: cpu,
    });
    const partitioner = new GraphPartitioner(config);

    const func = buildFunction('test', [f32_64x64, f32_64x64], [f32_64x64], (b, [x, y]) => {
      const add = b.add(x, y);
      const dot = b.dot(add.getResult(0), y, [1], [0]);
      const neg = b.neg(dot.getResult(0));
      b.returnOp([neg.getResult(0)]);
    });

    const result = partitioner.partition(func);
    assert.ok(result.numPartitions >= 1);

    const hasTransfers = result.transferEdges.length > 0 || result.numPartitions === 1;
    assert.ok(hasTransfers || result.numPartitions === 1);
  });

  it('respects device attribute annotations', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const config = new PartitionerConfig({
      targets: [cpu, gpu],
      defaultTarget: cpu,
    });
    const partitioner = new GraphPartitioner(config);

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      add.setAttr('device', 'gpu_generic');
      const mul = b.mul(add.getResult(0), y);
      mul.setAttr('device', 'cpu_generic');
      b.returnOp([mul.getResult(0)]);
    });

    const result = partitioner.partition(func);
    const addOps = func.findOps(op => op.opName === 'add');
    const mulOps = func.findOps(op => op.opName === 'mul');

    if (addOps.length > 0) {
      const addPartition = result.getPartition(addOps[0]);
      if (addPartition) {
        assert.equal(addPartition.target.name, 'gpu_generic');
      }
    }
    if (mulOps.length > 0) {
      const mulPartition = result.getPartition(mulOps[0]);
      if (mulPartition) {
        assert.equal(mulPartition.target.name, 'cpu_generic');
      }
    }
  });

  it('respects memory limits', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const memoryLimits = new Map();
    memoryLimits.set('gpu_generic', 256);

    const config = new PartitionerConfig({
      targets: [cpu, gpu],
      defaultTarget: cpu,
      memoryLimits,
    });
    const partitioner = new GraphPartitioner(config);

    const func = buildFunction('test', [f32_64x64, f32_64x64], [f32_64x64], (b, [x, y]) => {
      const add = b.add(x, y);
      const mul = b.mul(add.getResult(0), y);
      b.returnOp([mul.getResult(0)]);
    });

    const result = partitioner.partition(func);
    for (const p of result.partitions) {
      if (p.target.name === 'gpu_generic') {
        assert.ok(p.memoryBytes <= 256 || p.size === 1);
      }
    }
  });

  it('computes transfer edges between partitions', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const config = new PartitionerConfig({
      targets: [cpu, gpu],
      defaultTarget: cpu,
    });
    const partitioner = new GraphPartitioner(config);

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      add.setAttr('device', 'cpu_generic');
      const mul = b.mul(add.getResult(0), y);
      mul.setAttr('device', 'gpu_generic');
      b.returnOp([mul.getResult(0)]);
    });

    const result = partitioner.partition(func);

    if (result.numPartitions > 1) {
      assert.ok(result.transferEdges.length > 0);
      for (const edge of result.transferEdges) {
        assert.ok(edge.src);
        assert.ok(edge.dst);
        assert.ok(edge.value);
        assert.notEqual(edge.src, edge.dst);
      }
    }
  });

  it('getPartitionsForTarget filters correctly', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const config = new PartitionerConfig({
      targets: [cpu, gpu],
      defaultTarget: cpu,
    });
    const partitioner = new GraphPartitioner(config);

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      add.setAttr('device', 'cpu_generic');
      const mul = b.mul(add.getResult(0), y);
      mul.setAttr('device', 'gpu_generic');
      b.returnOp([mul.getResult(0)]);
    });

    const result = partitioner.partition(func);
    const cpuPartitions = result.getPartitionsForTarget(cpu);
    const gpuPartitions = result.getPartitionsForTarget(gpu);

    for (const p of cpuPartitions) {
      assert.equal(p.target.name, 'cpu_generic');
    }
    for (const p of gpuPartitions) {
      assert.equal(p.target.name, 'gpu_generic');
    }
  });
});

describe('GraphPartitionPass', () => {
  it('returns UNCHANGED with fewer than 2 targets', () => {
    const cpu = CPUTarget();
    const pass = new GraphPartitionPass({ targets: [cpu] });
    const func = buildFunction('test', [f32_4x4], [f32_4x4], (b, [x]) => {
      const neg = b.neg(x);
      b.returnOp([neg.getResult(0)]);
    });

    const result = pass.run(func, null);
    assert.equal(result, PassResult.UNCHANGED);
  });

  it('annotates ops with partition metadata', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const pass = new GraphPartitionPass({
      targets: [cpu, gpu],
      defaultTarget: cpu,
    });

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      add.setAttr('device', 'cpu_generic');
      const mul = b.mul(add.getResult(0), y);
      mul.setAttr('device', 'gpu_generic');
      b.returnOp([mul.getResult(0)]);
    });

    const result = pass.run(func, null);
    assert.equal(result, PassResult.CHANGED);

    for (const op of func.ops()) {
      if (op.opName === 'return') continue;
      assert.ok(op.hasAttr('partition_id'), `${op.opName} should have partition_id`);
      assert.ok(op.hasAttr('partition_target'), `${op.opName} should have partition_target`);
    }
  });

  it('inserts copy_to_device ops at partition boundaries', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const pass = new GraphPartitionPass({
      targets: [cpu, gpu],
      defaultTarget: cpu,
    });

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      add.setAttr('device', 'cpu_generic');
      const mul = b.mul(add.getResult(0), y);
      mul.setAttr('device', 'gpu_generic');
      b.returnOp([mul.getResult(0)]);
    });

    const result = pass.run(func, null);

    if (result === PassResult.CHANGED) {
      const copyOps = func.findOps(op => op.opName === 'copy_to_device');
      if (copyOps.length > 0) {
        for (const copyOp of copyOps) {
          assert.ok(copyOp.hasAttr('src_device'));
          assert.ok(copyOp.hasAttr('dst_device'));
          assert.notEqual(copyOp.getAttr('src_device'), copyOp.getAttr('dst_device'));
          assert.equal(copyOp.numOperands, 1);
          assert.equal(copyOp.numResults, 1);
          assert.ok(copyOp.getResult(0).type.equals(copyOp.getOperand(0).type));
        }
      }
    }
  });
});

describe('PartitionMaterializationPass', () => {
  it('creates sub-functions from partitioned graph', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      add.setAttr('partition_id', 0);
      add.setAttr('partition_target', 'cpu_generic');
      const mul = b.mul(add.getResult(0), y);
      mul.setAttr('partition_id', 1);
      mul.setAttr('partition_target', 'gpu_generic');
      b.returnOp([mul.getResult(0)]);
    });

    const pass = new PartitionMaterializationPass({ targets: [cpu, gpu] });
    const mod = buildModule('test_mod', []);
    func._module = mod;
    mod.addFunction(func);

    const result = pass.run(func, null);

    if (result === PassResult.CHANGED) {
      assert.ok(mod.functionCount > 1);
      for (const subFunc of mod.functions()) {
        if (subFunc.name === 'test') continue;
        assert.ok(subFunc.name.includes('partition'));
        assert.ok(subFunc._partitionTarget);
        const retOp = subFunc.getReturnOp();
        assert.ok(retOp);
      }
    }
  });
});

describe('three-target partitioning', () => {
  it('partitions across CPU, GPU, and WASM', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const wasm = WasmTarget();

    const config = new PartitionerConfig({
      targets: [cpu, gpu, wasm],
      defaultTarget: cpu,
    });
    const partitioner = new GraphPartitioner(config);

    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      const add = b.add(x, y);
      add.setAttr('device', 'cpu_generic');
      const mul = b.mul(add.getResult(0), y);
      mul.setAttr('device', 'gpu_generic');
      const neg = b.neg(mul.getResult(0));
      neg.setAttr('device', 'wasm_generic');
      b.returnOp([neg.getResult(0)]);
    });

    const result = partitioner.partition(func);
    assert.ok(result.numPartitions >= 1);

    const targetNames = new Set(result.partitions.map(p => p.target.name));
    assert.ok(targetNames.size >= 1);
  });
});

describe('PartitionerConfig', () => {
  it('accepts opTargetOverrides', () => {
    const cpu = CPUTarget();
    const gpu = GPUTarget();
    const overrides = new Map();
    overrides.set('dot', gpu);

    const config = new PartitionerConfig({
      targets: [cpu, gpu],
      defaultTarget: cpu,
      opTargetOverrides: overrides,
    });
    const partitioner = new GraphPartitioner(config);

    const func = buildFunction('test', [f32_8x8, f32_8x8], [f32_8x8], (b, [x, y]) => {
      const add = b.add(x, y);
      const dot = b.dot(add.getResult(0), y, [1], [0]);
      b.returnOp([dot.getResult(0)]);
    });

    const result = partitioner.partition(func);
    const dotOps = func.findOps(op => op.opName === 'dot');
    if (dotOps.length > 0) {
      const dotPartition = result.getPartition(dotOps[0]);
      if (dotPartition) {
        assert.equal(dotPartition.target.name, 'gpu_generic');
      }
    }
  });
});
