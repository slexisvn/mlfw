import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';
import { TensorType, dimEquals } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { explainer } from '../explain.js';
import { TraceLevel } from '../../support/trace.js';
import type { Block } from '../../ir/graph/block.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export function broadcastDimsFor(operand: TensorType, result: TensorType): number[] | null {
  const offset = result.shape.length - operand.shape.length;
  if (offset < 0) return null;
  const dims: number[] = [];
  let needed = offset > 0;
  for (let i = 0; i < operand.shape.length; i++) {
    if (!dimEquals(operand.shape[i], result.shape[i + offset])) {
      if (operand.shape[i] !== 1) return null;
      needed = true;
    }
    dims.push(i + offset);
  }
  return needed ? dims : null;
}

export class ExplicitBroadcastPass extends FunctionPass {
  constructor() {
    super('ExplicitBroadcastPass');
  }

  override run(func: PassTarget): PassResultValue {
    const targets: Operation[] = [];
    for (const op of (func as GraphFunction).opsRecursive()) {
      const def = registry.get(op.opName);
      if (def && def.isElementwise && op.numResults === 1) targets.push(op);
    }

    const builder = new IRBuilder(func as GraphFunction);
    const explain = explainer(this.trace, this.name);
    let inserted = 0;

    for (const op of targets) {
      const result = op.getResult(0).type;
      if (!(result instanceof TensorType) || !op.parentBlock) continue;
      for (let i = 0; i < op.numOperands; i++) {
        const operand = op.getOperand(i).type;
        if (!(operand instanceof TensorType)) continue;
        const dims = broadcastDimsFor(operand, result);
        if (!dims) continue;
        builder.block = op.parentBlock as Block;
        builder.setInsertionPoint(op);
        const wide = builder.withLocation(op.loc,
          () => builder.broadcast(op.getOperand(i), result.shape, dims, [], op.getResult(0)));
        op.replaceOperand(i, wide.getResult(0));
        inserted++;
        if (explain) {
          explain(op.opName, 'operand broadcast made explicit',
            'the op broadcasts this operand implicitly, which no shape-checked consumer of the IR can see',
            { operand: i, from: operand.shape, to: result.shape, broadcastDimensions: dims });
        }
      }
    }

    builder.setInsertionPointToEnd();
    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        broadcastsInserted: inserted, level: TraceLevel.DEBUG,
      });
    }
    return inserted > 0 ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
