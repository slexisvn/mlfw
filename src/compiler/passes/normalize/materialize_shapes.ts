import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';
import { TensorType } from '../../ir/graph/types.js';
import { sharesOperandAndResultShape } from '../../ir/graph/op_traits.js';
import { sizesOperandSpan, dynamicResultExtents } from '../../ir/graph/mlir_format.js';
import { explainer } from '../explain.js';
import { TraceLevel } from '../../support/trace.js';
import type { Block } from '../../ir/graph/block.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { PassResultValue, PassTarget } from '../pass.js';

class BlockOrder {
  private readonly _orders: Map<Block, Map<Operation, number>>;

  constructor() {
    this._orders = new Map();
  }

  of(block: Block): Map<Operation, number> {
    let order = this._orders.get(block);
    if (!order) {
      order = new Map<Operation, number>();
      let index = 0;
      for (const op of block) order.set(op, index++);
      this._orders.set(block, order);
    }
    return order;
  }
}

function extentSource(op: Operation, axis: number, order: Map<Operation, number>): [Value, number] | null {
  const result = op.getResult(0);
  const position = order.get(op) as number;
  for (const use of result.uses()) {
    if (!sharesOperandAndResultShape(use.user.opName)) continue;
    for (const operand of use.user.operands) {
      if (operand === result) continue;
      const type = operand.type;
      if (!(type instanceof TensorType) || type.rank !== (result.type as TensorType).rank) continue;
      const producer = operand.definingOp;
      if (producer && (order.get(producer) ?? Infinity) >= position) continue;
      return [operand, axis];
    }
  }
  return null;
}

function extentBySymbol(op: Operation, result: number, axis: number): [Value, number] | null {
  const symbols = op.getResult(result).symbolicShape;
  const symbol = symbols ? symbols[axis] : null;
  if (typeof symbol !== 'string') return null;
  for (const operand of op.operands) {
    const operandSymbols = operand.symbolicShape;
    if (!operandSymbols) continue;
    for (let i = 0; i < operandSymbols.length; i++) {
      if (operandSymbols[i] === symbol) return [operand, i];
    }
  }
  return null;
}

export class MaterializeShapesPass extends FunctionPass {
  constructor() {
    super('MaterializeShapesPass');
  }

  override run(func: PassTarget): PassResultValue {
    const targets: Operation[] = [];
    for (const op of (func as GraphFunction).opsRecursive()) {
      const sizes = sizesOperandSpan(op);
      if (!sizes || sizes.count > 0) continue;
      if (dynamicResultExtents(op).length > 0) targets.push(op);
    }
    if (targets.length === 0) return PassResult.UNCHANGED;

    const builder = new IRBuilder(func as GraphFunction);
    const explain = explainer(this.trace, this.name);
    const orders = new BlockOrder();
    let materialised = 0;

    for (const op of targets) {
      const block = op.parentBlock as Block;
      const dynamic = dynamicResultExtents(op);
      const extents: Value[] = [];
      const asked = new Map<string, Value>();
      for (const { result, axis } of dynamic) {
        const bySymbol = extentBySymbol(op, result, axis);
        const source = bySymbol ?? (result === 0 ? extentSource(op, axis, orders.of(block)) : null);
        if (!source) {
          throw new Error(
            `${this.name}: '${op.opName}' has a dynamic extent on result ${result} axis ${axis} that nothing in scope carries`);
        }
        const key = `${source[0].id}:${source[1]}`;
        let extent = asked.get(key);
        if (!extent) {
          builder.block = block;
          builder.setInsertionPoint(op);
          extent = builder.withLocation(op.loc, () => builder.dim(source[0], source[1])).getResult(0);
          asked.set(key, extent);
        }
        extents.push(extent);
      }
      for (const extent of extents) op.appendOperand(extent);
      materialised += extents.length;
      if (explain) {
        explain(op.opName, 'shape made explicit',
          'the op decides a result shape rather than inheriting one, so a dynamic extent of it is a number no operand type carries',
          { extents: dynamic });
      }
    }

    builder.setInsertionPointToEnd();
    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        extentsMaterialised: materialised, level: TraceLevel.DEBUG,
      });
    }
    return PassResult.CHANGED;
  }
}
