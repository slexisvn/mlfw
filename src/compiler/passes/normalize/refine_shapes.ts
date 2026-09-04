import { FunctionPass, PassResult } from '../pass.js';
import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { mlirFormOfOp, sizesOperandSpan, derivedAttrValue, dynamicResultExtents } from '../../ir/graph/mlir_format.js';
import { explainer } from '../explain.js';
import { TraceLevel } from '../../support/trace.js';
import type { Block } from '../../ir/graph/block.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { Dim, IRType, Shape } from '../../ir/graph/types.js';
import type { PassResultValue, PassTarget } from '../pass.js';

function isDynamic(type: IRType | null | undefined): boolean {
  return type instanceof TensorType && type.shape.some((d) => d === DYNAMIC);
}

function staticIntOf(value: Value): number | null {
  const op = value.definingOp;
  if (!op) return null;
  if (op.opName === 'dim') {
    const source = op.getOperand(0).type;
    const axis = op.getAttr<number>('dimension');
    if (!(source instanceof TensorType) || typeof axis !== 'number') return null;
    const extent = source.shape[axis];
    return typeof extent === 'number' && extent >= 0 ? extent : null;
  }
  if (op.opName === 'constant') {
    const type = op.getResult(0).type;
    const value_ = op.getAttr('value');
    if (!(type instanceof TensorType) || type.rank !== 0) return null;
    return typeof value_ === 'number' && Number.isInteger(value_) ? value_ : null;
  }
  return null;
}

export class ShapeRefinementPass extends FunctionPass {
  private readonly _argShapes: readonly Shape[];
  private _refined: number;

  constructor(argShapes: readonly Shape[]) {
    super('ShapeRefinementPass');
    this._argShapes = argShapes;
    this._refined = 0;
  }

  override run(target: PassTarget): PassResultValue {
    const func = target as GraphFunction;
    if (this._argShapes.length !== func.inputTypes.length) {
      throw new Error(
        `${this.name}: '${func.name}' takes ${func.inputTypes.length} arguments but ${this._argShapes.length} shapes were given`);
    }
    const argTypes = func.inputTypes.map((type, i) => {
      if (!(type instanceof TensorType)) return type;
      const shape = this._argShapes[i];
      if (shape.length !== type.rank) {
        throw new Error(
          `${this.name}: argument ${i} of '${func.name}' has rank ${type.rank} but the shape given has rank ${shape.length}`);
      }
      for (let axis = 0; axis < shape.length; axis++) {
        const declared = type.shape[axis];
        if (declared !== DYNAMIC && declared !== shape[axis]) {
          throw new Error(
            `${this.name}: argument ${i} of '${func.name}' is ${declared} on axis ${axis}, which the shape given contradicts with ${shape[axis]}`);
        }
      }
      return type.withShape(shape);
    });
    if (!func.inputTypes.some(isDynamic) && !func.outputTypes.some(isDynamic)) {
      return PassResult.UNCHANGED;
    }

    this._refined = 0;
    this._refineBlock(func.entryBlock, argTypes);
    func.inputTypes = Object.freeze([...argTypes]);

    const terminator = func.entryBlock.lastOp;
    if (terminator && terminator.isTerminator()) {
      func.outputTypes = Object.freeze(terminator.operands.map((v) => v.type));
    }
    const stillDynamic = [...func.inputTypes, ...func.outputTypes].some(isDynamic);
    if (stillDynamic) {
      throw new Error(
        `${this.name}: '${func.name}' still has a dynamic extent on its boundary after refinement`);
    }
    func.bumpVersion();

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        typesRefined: this._refined, level: TraceLevel.DEBUG,
      });
    }
    return PassResult.CHANGED;
  }

  private _refineBlock(block: Block, argTypes: readonly IRType[] | null): void {
    if (argTypes) {
      if (block.arguments.length !== argTypes.length) {
        throw new Error(
          `${this.name}: a block takes ${block.arguments.length} arguments but its op provides ${argTypes.length}`);
      }
      for (let i = 0; i < block.arguments.length; i++) block.arguments[i].type = argTypes[i];
    }
    for (const op of [...block.ops()]) this._refineOp(op);
  }

  private _refineOp(op: Operation): void {
    const def = op.def;
    if (!def) return;

    if (op.regions.length > 0) {
      const argTypes = def.inferRegionArgTypes ? def.inferRegionArgTypes(op) : null;
      for (let r = 0; r < op.regions.length; r++) {
        const region = op.regions[r];
        for (const block of region.blocks) {
          this._refineBlock(block, argTypes && block === region.entryBlock ? argTypes[r] : null);
        }
      }
      const fromRegions = def.inferResultTypesFromRegions ? def.inferResultTypesFromRegions(op) : null;
      if (fromRegions) {
        this._assign(op, fromRegions);
        this._applySizes(op);
        return;
      }
    }

    if (!def.inferResultTypes) return;
    const types = def.inferResultTypes(
      op.operands.map((v) => v.type), op.attributes, op.results.map((r) => r.type));
    if (types) this._assign(op, types);
    this._applySizes(op);
  }

  private _assign(op: Operation, types: readonly IRType[]): void {
    for (let i = 0; i < op.numResults && i < types.length; i++) {
      const result = op.getResult(i);
      if (result.type === types[i]) continue;
      result.type = types[i];
      this._refined++;
    }
  }

  private _applySizes(op: Operation): void {
    const form = mlirFormOfOp(op.opName);
    const sizes = sizesOperandSpan(op);
    if (!form || !sizes || sizes.count === 0) return;
    const start = sizes.start;

    const dynamic = dynamicResultExtents(op);
    if (dynamic.length > 0) {
      if (dynamic.length !== sizes.count) return;

      const extents = op.operands.slice(start).map(staticIntOf);
      if (extents.some((extent) => extent === null)) return;

      const shapes = new Map<number, Dim[]>();
      for (let i = 0; i < dynamic.length; i++) {
        const { result, axis } = dynamic[i];
        let shape = shapes.get(result);
        if (!shape) {
          shape = [...(op.getResult(result).type as TensorType).shape];
          shapes.set(result, shape);
        }
        shape[axis] = extents[i] as number;
      }
      for (const [result, shape] of shapes) {
        const refined = (op.getResult(result).type as TensorType).withShape(shape);
        op.getResult(result).type = refined;
        this._refined++;
      }
    }

    for (const derived of form.derived) {
      const value = derivedAttrValue(derived.from, op.getResult(0).type);
      if (value !== undefined) op.setAttr(derived.ir, value);
    }

    const sources = op.operands.slice(start).map((v) => v.definingOp);
    op.truncateOperands(start);
    for (const source of sources) {
      if (source && source.parentBlock && source.results.every((r) => !r.hasUses)) source.erase();
    }

    const explain = explainer(this.trace, this.name);
    if (explain) {
      explain(op.opName, 'shape made static',
        'the extents the op was handed as values are known here, so they belong in its type',
        { extents: dynamic });
    }
  }
}
