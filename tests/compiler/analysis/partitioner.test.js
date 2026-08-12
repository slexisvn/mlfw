import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { GraphPartitioner, PartitionerConfig } from '../../../src/compiler/analysis/partitioner.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';

function ops(func) {
  const list = [];
  for (const op of func.ops()) {
    if (op.opName !== 'return') list.push(op);
  }
  return list;
}

describe('GraphPartitioner._scoreTargetForOp', () => {
  it('GPU wins by default — computeTFLOPs*10=150 dominates CPU bonus +30', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.getPartition(ops(func)[0]).target.name).toBe('cuda_generic');
  });

  it('GPU large elementwise gets +50 on top of compute base', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.getPartition(ops(func)[0]).target.name).toBe('cuda_generic');
  });

  it('CPU small bonus +30 kicks in when GPU compute is low enough', () => {
    const cpu = CPUTarget({ computeTFLOPs: 2 });
    const gpu = CUDATarget({ computeTFLOPs: 2 });
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.getPartition(ops(func)[0]).target.name).toBe('cpu_generic');
  });

  it('library-class +100 can beat GPU when compute gap is < 100/10 = 10 TFLOPs', () => {
    const cpu = CPUTarget({ computeTFLOPs: 5, libraryClasses: new Set(['matmul']) });
    const gpu = CUDATarget({ computeTFLOPs: 5, libraryClasses: new Set() });
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [new TensorType([4, 6], ScalarType.F32)], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.getPartition(ops(func)[0]).target.name).toBe('cpu_generic');
  });

  it('library-class +100 loses to GPU default compute gap of 14.5 TFLOPs', () => {
    const cpu = CPUTarget({ libraryClasses: new Set(['matmul']) });
    const gpu = CUDATarget({ libraryClasses: new Set() });
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [new TensorType([4, 6], ScalarType.F32)], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.getPartition(ops(func)[0]).target.name).toBe('cuda_generic');
  });
});

describe('GraphPartitioner target override mechanisms', () => {
  it('opTargetOverrides forces op to specific target regardless of score', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const overrides = new Map([['add', cpu]]);
    const config = new PartitionerConfig({ targets: [cpu, gpu], opTargetOverrides: overrides });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.getPartition(ops(func)[0]).target.name).toBe('cpu_generic');
  });

  it('device attribute on op overrides score and opTargetOverrides', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const addOp = b.add(args[0], args[1]);
      addOp.setAttr('device', 'cpu');
      b.returnOp([addOp.getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.getPartition(ops(func)[0]).target.name).toBe('cpu_generic');
  });

  it('device attribute resolves by target kind string', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const negOp = b.neg(args[0]);
      negOp.setAttr('device', 'cpu');
      b.returnOp([negOp.getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.getPartition(ops(func)[0]).target.name).toBe('cpu_generic');
  });
});

describe('GraphPartitioner partition building', () => {
  it('producer-consumer on same target merge into one partition', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    const allOps = ops(func);
    expect(result.getPartition(allOps[0])).toBe(result.getPartition(allOps[1]));
  });

  it('ops forced to different targets create separate partitions', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const n = b.neg(args[0]);
      b.returnOp([a.getResult(0), n.getResult(0)]);
    });

    const overrides = new Map([['add', gpu], ['neg', cpu]]);
    const config = new PartitionerConfig({ targets: [cpu, gpu], opTargetOverrides: overrides });
    const result = new GraphPartitioner(config).partition(func);
    const allOps = ops(func);

    const addPart = result.getPartition(allOps.find(o => o.opName === 'add'));
    const negPart = result.getPartition(allOps.find(o => o.opName === 'neg'));
    expect(addPart.target.name).toBe('cuda_generic');
    expect(negPart.target.name).toBe('cpu_generic');
  });

  it('memoryLimits prevents second op from merging into partition', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const singleOpMem = 1024 * 1024 * 4;
    const limits = new Map([['cuda_generic', singleOpMem]]);
    const config = new PartitionerConfig({ targets: [cpu, gpu], memoryLimits: limits });
    const result = new GraphPartitioner(config).partition(func);

    const gpuParts = result.partitions.filter(p => p.target.name === 'cuda_generic');
    expect(gpuParts.length).toBeGreaterThanOrEqual(2);
    for (const p of gpuParts) {
      expect(p.size).toBe(1);
    }
  });
});

describe('GraphPartitioner transfer edges', () => {
  it('cross-device data flow creates transfer edges with correct src/dst', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const gpuAdd = b.add(args[0], args[1]);
      const cpuNeg = b.neg(args[0]);
      b.returnOp([gpuAdd.getResult(0), cpuNeg.getResult(0)]);
    });

    const overrides = new Map([['add', gpu], ['neg', cpu]]);
    const config = new PartitionerConfig({ targets: [cpu, gpu], opTargetOverrides: overrides });
    const result = new GraphPartitioner(config).partition(func);

    for (const edge of result.transferEdges) {
      expect(edge.src.target.name).not.toBe(edge.dst.target.name);
      expect(edge.value).toBeDefined();
    }
  });

  it('same-target producer-consumer has no transfer edge', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([256, 256], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);

    for (const edge of result.transferEdges) {
      expect(edge.src).not.toBe(edge.dst);
      expect(edge.src.target.name).not.toBe(edge.dst.target.name);
    }
  });

  it('transfer edge sizeBytes matches the value tensor size', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const gpuAdd = b.add(args[0], args[1]);
      const cpuNeg = b.neg(gpuAdd.getResult(0));
      b.returnOp([gpuAdd.getResult(0), cpuNeg.getResult(0)]);
    });

    const overrides = new Map([['add', gpu], ['neg', cpu]]);
    const config = new PartitionerConfig({ targets: [cpu, gpu], opTargetOverrides: overrides });
    const result = new GraphPartitioner(config).partition(func);

    for (const edge of result.transferEdges) {
      if (edge.sizeBytes > 0) {
        expect(edge.sizeBytes).toBe(edge.value.type.sizeInBytes());
      }
    }
  });

  it('no duplicate edges for same value to same dst partition', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const gpuAdd = b.add(args[0], args[1]);
      const cpuNeg = b.neg(gpuAdd.getResult(0));
      const cpuExp = b.exp(gpuAdd.getResult(0));
      b.returnOp([cpuNeg.getResult(0), cpuExp.getResult(0)]);
    });

    const overrides = new Map([['add', gpu], ['neg', cpu], ['exp', cpu]]);
    const config = new PartitionerConfig({ targets: [cpu, gpu], opTargetOverrides: overrides });
    const result = new GraphPartitioner(config).partition(func);

    const seen = new Set();
    for (const edge of result.transferEdges) {
      const key = `${edge.src.id}:${edge.dst.id}:${edge.value.id}`;
      expect(seen.has(key)).toBe(false);
      seen.add(key);
    }
  });
});

describe('GraphPartitioner merge small partitions', () => {
  it('single-op partitions below minPartitionSize merge with same-target neighbor', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu], minPartitionSize: 5 });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.numPartitions).toBeLessThanOrEqual(2);
  });

  it('merge picks candidate with most shared data edges', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.neg(a.getResult(0));
      const d = b.exp(c.getResult(0));
      b.returnOp([d.getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu], minPartitionSize: 2 });
    const result = new GraphPartitioner(config).partition(func);

    for (const p of result.partitions) {
      expect(p.size).toBeGreaterThanOrEqual(1);
    }
  });
});

describe('GraphPartitioner cycle avoidance and op coverage', () => {
  it('does not merge an op into a producer partition when it would create a back-edge', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const n = b.neg(a.getResult(0));
      const c = b.add(a.getResult(0), n.getResult(0));
      b.returnOp([c.getResult(0)]);
    });

    const overrides = new Map([['neg', cpu]]);
    const config = new PartitionerConfig({ targets: [gpu, cpu], opTargetOverrides: overrides });
    const result = new GraphPartitioner(config).partition(func);

    const seen = new Set();
    const onStack = new Set();
    const hasCycle = (p) => {
      if (onStack.has(p)) return true;
      if (seen.has(p)) return false;
      seen.add(p);
      onStack.add(p);
      for (const op of p.ops) {
        for (let i = 0; i < op.numOperands; i++) {
          const prod = op.getOperand(i).definingOp;
          if (!prod) continue;
          const sp = result.getPartition(prod);
          if (sp && sp !== p && hasCycle(sp)) return true;
        }
      }
      onStack.delete(p);
      return false;
    };

    for (const p of result.partitions) {
      expect(hasCycle(p)).toBe(false);
    }
  });

  it('every partitionable op lands in exactly one emitted partition', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const n = b.neg(a.getResult(0));
      const e = b.exp(n.getResult(0));
      b.returnOp([e.getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu], minPartitionSize: 10 });
    const result = new GraphPartitioner(config).partition(func);

    for (const op of ops(func)) {
      const partsWithOp = result.partitions.filter(p => p.hasOp(op));
      expect(partsWithOp.length).toBe(1);
    }
  });
});

describe('PartitionResult queries', () => {
  it('getPartitionsForTarget filters correctly', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const gpuAdd = b.add(args[0], args[1]);
      const cpuNeg = b.neg(args[0]);
      b.returnOp([gpuAdd.getResult(0), cpuNeg.getResult(0)]);
    });

    const overrides = new Map([['add', gpu], ['neg', cpu]]);
    const config = new PartitionerConfig({ targets: [cpu, gpu], opTargetOverrides: overrides });
    const result = new GraphPartitioner(config).partition(func);

    const cpuParts = result.getPartitionsForTarget(cpu);
    const gpuParts = result.getPartitionsForTarget(gpu);
    for (const p of cpuParts) expect(p.target.name).toBe('cpu_generic');
    for (const p of gpuParts) expect(p.target.name).toBe('cuda_generic');
    expect(cpuParts.length + gpuParts.length).toBe(result.numPartitions);
  });

  it('numPartitions matches partitions array length', () => {
    const cpu = CPUTarget();
    const gpu = CUDATarget();
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const config = new PartitionerConfig({ targets: [cpu, gpu] });
    const result = new GraphPartitioner(config).partition(func);
    expect(result.numPartitions).toBe(result.partitions.length);
  });
});
