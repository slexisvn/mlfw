import { registry } from '../ir/graph/ops.js';
import { DYNAMIC } from '../ir/graph/types.js';
import { SymInt } from './sym_int.js';

export class ShapeResult {
  constructor(shapes) {
    this.shapes = shapes;
  }

  shapeOf(value) {
    return this.shapes.get(value) || null;
  }
}

export class ShapeAnalysis {
  static get name() { return 'shape'; }
  static get depKey() { return 'shape'; }
  static get dependencies() { return []; }

  static compute(func) {
    const shapes = new Map();
    let nextDynId = 0;

    const resolveShape = (shape) => {
      if (!shape) return null;
      return shape.map(d => {
        if (d === DYNAMIC) {
          return SymInt.var(`d${nextDynId++}`);
        }
        return d;
      });
    };

    for (const arg of func.args) {
      shapes.set(arg, resolveShape(arg.type.shape));
    }

    for (const op of func.ops()) {
      const def = registry.get(op.opName);

      if (def && def.propagateSymbolicShapes) {
        const resShapes = def.propagateSymbolicShapes(op, shapes);
        if (resShapes) {
          for (let i = 0; i < Math.min(resShapes.length, op.numResults); i++) {
            if (resShapes[i]) {
              shapes.set(op.getResult(i), resShapes[i]);
            }
          }
          continue;
        }
      }

      if (def && def.inferResultTypes) {
        const operandTypes = op.operands.map(o => o.type);
        const resultTypes = def.inferResultTypes(
          operandTypes, op.attributes, op.results.map(r => r.type)
        );

        if (resultTypes) {
          for (let i = 0; i < Math.min(resultTypes.length, op.numResults); i++) {
            shapes.set(op.getResult(i), resolveShape(resultTypes[i].shape));
          }
          continue;
        }
      }

      for (let i = 0; i < op.numResults; i++) {
        shapes.set(op.getResult(i), resolveShape(op.getResult(i).type.shape));
      }
    }

    return new ShapeResult(shapes);
  }
}
