import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';

import { GraphFunction } from '../../ir/graph/function.js';
import { GraphPartitioner, PartitionerConfig } from '../../analysis/partitioner.js';
import { topoSortOps, computePartitionIO } from './partition_core.js';
import { TraceLevel } from '../../pipeline/trace.js';

export class GraphPartitionPass extends FunctionPass {
  constructor(config = {}) {
    super('GraphPartitionPass');
    this.partitionerConfig = new PartitionerConfig(config);
    this.partitionResult = null;
  }

  run(func, analysisManager) {
    if (this.partitionerConfig.targets.length < 2) return PassResult.UNCHANGED;

    const partitioner = new GraphPartitioner(this.partitionerConfig);
    this.partitionResult = partitioner.partition(func);

    if (this.partitionResult.numPartitions <= 1) return PassResult.UNCHANGED;

    this._annotateOps();
    this._insertTransferOps(func);

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        numPartitions: this.partitionResult.numPartitions,
        transferEdges: this.partitionResult.transferEdges.length,
        level: TraceLevel.DEBUG,
      });
    }

    func.bumpVersion();
    return PassResult.CHANGED;
  }

  _annotateOps() {
    for (const partition of this.partitionResult.partitions) {
      for (const op of partition.ops) {
        op.setAttr('partition_id', partition.id);
        op.setAttr('partition_target', partition.target.name);
      }
    }
  }

  _insertTransferOps(func) {
    const block = func.entryBlock;
    const order = this._buildOrderIndex(block);
    const { useMap, firstInPart } = this._buildInsertionIndex(block);

    for (const edge of this.partitionResult.transferEdges) {
      const value = edge.value;
      const srcDevice = edge.src.target.name;
      const dstDevice = edge.dst.target.name;

      if (srcDevice === dstDevice) continue;

      const copyOp = new Operation(
        'copy_to_device',
        [value],
        [value.type],
        { src_device: srcDevice, dst_device: dstDevice }
      );
      copyOp.setAttr('partition_id', edge.dst.id);
      copyOp.setAttr('partition_target', dstDevice);

      const partUses = useMap.get(edge.dst);
      const firstUse = (partUses && partUses.get(value)) || firstInPart.get(edge.dst) || null;
      const defOp = value.definingOp;

      if (firstUse && defOp && order.has(defOp) && order.has(firstUse) &&
          order.get(defOp) >= order.get(firstUse)) {
        block.insertAfter(copyOp, defOp);
      } else if (firstUse) {
        block.insertBefore(copyOp, firstUse);
      } else if (defOp && order.has(defOp)) {
        block.insertAfter(copyOp, defOp);
      } else {
        const returnOp = func.getReturnOp();
        if (returnOp) {
          block.insertBefore(copyOp, returnOp);
        } else {
          block.pushOp(copyOp);
        }
      }

      const copyResult = copyOp.getResult(0);
      for (const op of edge.dst.ops) {
        for (let i = 0; i < op.numOperands; i++) {
          if (op.getOperand(i) === value) {
            op.replaceOperand(i, copyResult);
          }
        }
      }
    }
  }

  _buildInsertionIndex(block) {
    const dstParts = [...new Set(this.partitionResult.transferEdges.map(e => e.dst))];
    const useMap = new Map();
    const firstInPart = new Map();
    for (const p of dstParts) useMap.set(p, new Map());

    for (const op of block.ops()) {
      for (const p of dstParts) {
        if (!p.hasOp(op)) continue;
        if (!firstInPart.has(p)) firstInPart.set(p, op);
        const uses = useMap.get(p);
        for (let i = 0; i < op.numOperands; i++) {
          const v = op.getOperand(i);
          if (!uses.has(v)) uses.set(v, op);
        }
      }
    }
    return { useMap, firstInPart };
  }

  _buildOrderIndex(block) {
    const order = new Map();
    let idx = 0;
    for (const op of block.ops()) order.set(op, idx++);
    return order;
  }
}

export class PartitionMaterializationPass extends FunctionPass {
  constructor(config = {}) {
    super('PartitionMaterializationPass');
    this.targets = config.targets || [];
  }

  run(func, analysisManager) {
    const partitionMap = this._collectPartitions(func);
    if (partitionMap.size <= 1) return PassResult.UNCHANGED;

    const module = this._getModule(func);
    if (!module) return PassResult.UNCHANGED;

    const subFunctions = this._materializePartitions(func, partitionMap);

    for (const subFunc of subFunctions) {
      module.addFunction(subFunc);
    }

    this._rewriteOriginalFunction(func, subFunctions, partitionMap);

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        partitions: partitionMap.size,
        subFunctions: subFunctions.length,
        level: TraceLevel.DEBUG,
      });
    }

    func.bumpVersion();
    return PassResult.CHANGED;
  }

  _collectPartitions(func) {
    const partitions = new Map();
    for (const op of func.ops()) {
      const pid = op.getAttr('partition_id');
      if (pid === undefined) continue;
      if (!partitions.has(pid)) {
        partitions.set(pid, {
          id: pid,
          target: op.getAttr('partition_target'),
          ops: [],
        });
      }
      partitions.get(pid).ops.push(op);
    }
    return partitions;
  }

  _getModule(func) {
    return func._module || null;
  }

  _materializePartitions(func, partitionMap) {
    const subFunctions = [];

    for (const [pid, partition] of partitionMap) {
      const opSet = new Set(partition.ops);
      const { inputs, outputs } = computePartitionIO(opSet, partition.ops);

      const inputTypes = inputs.map(v => v.type);
      const outputTypes = outputs.map(v => v.type);
      const subName = `${func.name}_partition_${pid}`;
      const subFunc = new GraphFunction(subName, inputTypes, outputTypes);
      subFunc._partitionTarget = partition.target;

      const valueMap = new Map();
      for (let i = 0; i < inputs.length; i++) {
        valueMap.set(inputs[i], subFunc.args[i]);
      }

      const sorted = topoSortOps(partition.ops);
      for (const op of sorted) {
        subFunc.entryBlock.pushOp(op.clone(valueMap));
      }

      const returnOperands = outputs.map(v => valueMap.get(v) || v);
      const returnOp = new Operation('return', returnOperands, []);
      subFunc.entryBlock.pushOp(returnOp);

      subFunctions.push(subFunc);
    }

    return subFunctions;
  }

  _rewriteOriginalFunction(func, subFunctions, partitionMap) {
    for (const subFunc of subFunctions) {
      func.setAttr?.(`sub_${subFunc.name}`, subFunc._partitionTarget);
    }
  }

}
