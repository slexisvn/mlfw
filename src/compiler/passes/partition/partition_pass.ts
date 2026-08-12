import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';

import { GraphFunction } from '../../ir/graph/function.js';
import { GraphPartitioner, PartitionerConfig } from '../../analysis/partitioner.js';
import { topoSortOps, computePartitionIO } from './partition_core.js';
import { TraceLevel } from '../../pipeline/trace.js';
import type { GraphModule } from '../../ir/graph/module.js';
import type { Block } from '../../ir/graph/block.js';
import type { Value } from '../../ir/graph/value.js';
import type { IRType } from '../../ir/graph/types.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';

type PartitionerOpts = ConstructorParameters<typeof PartitionerConfig>[0];
type PartitionResult = ReturnType<GraphPartitioner['partition']>;
type DevicePartition = PartitionResult['partitions'][number];
type MaterializedGroup = { id: number; target: string; ops: Operation[] };
export type PartitionMaterializationConfig = { targets?: readonly unknown[] };

export class GraphPartitionPass extends FunctionPass {
  partitionerConfig: PartitionerConfig;
  partitionResult: PartitionResult | null;

  constructor(config: PartitionerOpts = {}) {
    super('GraphPartitionPass');
    this.partitionerConfig = new PartitionerConfig(config);
    this.partitionResult = null;
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const graphFunc = func as GraphFunction;
    if (this.partitionerConfig.targets.length < 2) return PassResult.UNCHANGED;

    const partitioner = new GraphPartitioner(this.partitionerConfig);
    const result = partitioner.partition(graphFunc);
    this.partitionResult = result;

    if (result.numPartitions <= 1) return PassResult.UNCHANGED;

    this._annotateOps();
    this._insertTransferOps(graphFunc);

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        numPartitions: result.numPartitions,
        transferEdges: result.transferEdges.length,
        level: TraceLevel.DEBUG,
      });
    }

    graphFunc.bumpVersion();
    return PassResult.CHANGED;
  }

  _annotateOps(): void {
    for (const partition of (this.partitionResult as PartitionResult).partitions) {
      for (const op of partition.ops) {
        op.setAttr('partition_id', partition.id);
        op.setAttr('partition_target', partition.target.name);
      }
    }
  }

  _insertTransferOps(func: GraphFunction): void {
    const block = func.entryBlock as Block;
    const order = this._buildOrderIndex(block);
    const { useMap, firstInPart } = this._buildInsertionIndex(block);

    for (const edge of (this.partitionResult as PartitionResult).transferEdges) {
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
          (order.get(defOp) as number) >= (order.get(firstUse) as number)) {
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

  _buildInsertionIndex(block: Block): { useMap: Map<DevicePartition, Map<Value, Operation>>; firstInPart: Map<DevicePartition, Operation> } {
    const dstParts = [...new Set((this.partitionResult as PartitionResult).transferEdges.map(e => e.dst))];
    const useMap = new Map<DevicePartition, Map<Value, Operation>>();
    const firstInPart = new Map<DevicePartition, Operation>();
    for (const p of dstParts) useMap.set(p, new Map());

    for (const op of block.ops()) {
      for (const p of dstParts) {
        if (!p.hasOp(op)) continue;
        if (!firstInPart.has(p)) firstInPart.set(p, op);
        const uses = useMap.get(p) as Map<Value, Operation>;
        for (let i = 0; i < op.numOperands; i++) {
          const v = op.getOperand(i);
          if (!uses.has(v)) uses.set(v, op);
        }
      }
    }
    return { useMap, firstInPart };
  }

  _buildOrderIndex(block: Block): Map<Operation, number> {
    const order = new Map<Operation, number>();
    let idx = 0;
    for (const op of block.ops()) order.set(op, idx++);
    return order;
  }
}

export class PartitionMaterializationPass extends FunctionPass {
  targets: readonly unknown[];

  constructor(config: PartitionMaterializationConfig = {}) {
    super('PartitionMaterializationPass');
    this.targets = config.targets || [];
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const graphFunc = func as GraphFunction;
    const partitionMap = this._collectPartitions(graphFunc);
    if (partitionMap.size <= 1) return PassResult.UNCHANGED;

    const module = this._getModule(graphFunc);
    if (!module) return PassResult.UNCHANGED;

    const subFunctions = this._materializePartitions(graphFunc, partitionMap);

    for (const subFunc of subFunctions) {
      module.addFunction(subFunc);
    }

    this._rewriteOriginalFunction(graphFunc, subFunctions, partitionMap);

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        partitions: partitionMap.size,
        subFunctions: subFunctions.length,
        level: TraceLevel.DEBUG,
      });
    }

    graphFunc.bumpVersion();
    return PassResult.CHANGED;
  }

  _collectPartitions(func: GraphFunction): Map<number, MaterializedGroup> {
    const partitions = new Map<number, MaterializedGroup>();
    for (const op of func.ops()) {
      const pid = op.getAttr<number>('partition_id');
      if (pid === undefined) continue;
      if (!partitions.has(pid)) {
        partitions.set(pid, {
          id: pid,
          target: op.getAttr<string>('partition_target') as string,
          ops: [],
        });
      }
      (partitions.get(pid) as MaterializedGroup).ops.push(op);
    }
    return partitions;
  }

  _getModule(func: GraphFunction): GraphModule | null {
    return func._module || null;
  }

  _materializePartitions(func: GraphFunction, partitionMap: ReadonlyMap<number, MaterializedGroup>): GraphFunction[] {
    const subFunctions: GraphFunction[] = [];

    for (const [pid, partition] of partitionMap) {
      const opSet = new Set<Operation>(partition.ops);
      const { inputs, outputs } = computePartitionIO(opSet, partition.ops);

      const inputTypes: IRType[] = inputs.map(v => v.type);
      const outputTypes: IRType[] = outputs.map(v => v.type);
      const subName = `${func.name}_partition_${pid}`;
      const subFunc = new GraphFunction(subName, inputTypes, outputTypes);
      subFunc._partitionTarget = partition.target;

      const valueMap = new Map<Value, Value>();
      for (let i = 0; i < inputs.length; i++) {
        valueMap.set(inputs[i], subFunc.args[i]);
      }

      const entry = subFunc.entryBlock as Block;
      const sorted = topoSortOps(partition.ops);
      for (const op of sorted) {
        entry.pushOp(op.clone(valueMap));
      }

      const returnOperands = outputs.map(v => valueMap.get(v) || v);
      const returnOp = new Operation('return', returnOperands, []);
      entry.pushOp(returnOp);

      subFunctions.push(subFunc);
    }

    return subFunctions;
  }

  _rewriteOriginalFunction(func: GraphFunction, subFunctions: readonly GraphFunction[], partitionMap: ReadonlyMap<number, MaterializedGroup>): void {
    const attrHost = func as GraphFunction & { setAttr?(key: string, value: unknown): void };
    for (const subFunc of subFunctions) {
      attrHost.setAttr?.(`sub_${subFunc.name}`, subFunc._partitionTarget);
    }
  }

}
