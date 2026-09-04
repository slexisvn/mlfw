import { Block, Region } from './block.js';
import { topoSortByOperands } from './graph_algorithms.js';
import type { AttrValue, IRType } from './types.js';
import type { Operation } from './operation.js';
import type { Value } from './value.js';
import type { BlockArgument } from './value.js';
import type { GraphModule } from './module.js';

function* opsInRegions(op: Operation): Generator<Operation, void, undefined> {
  if (!op.regions || op.regions.length === 0) return;
  for (const region of op.regions) {
    for (const block of region.blocks) {
      for (const inner of block.ops()) {
        yield inner;
        yield* opsInRegions(inner);
      }
    }
  }
}

function* blocksInRegions(op: Operation): Generator<Block, void, undefined> {
  if (!op.regions || op.regions.length === 0) return;
  for (const region of op.regions) {
    for (const block of region.blocks) {
      yield block;
      for (const inner of block.ops()) yield* blocksInRegions(inner);
    }
  }
}

export class GraphFunction {
  declare _module?: GraphModule;
  declare _partitionTarget?: string;
  name: string;
  inputTypes: readonly IRType[];
  outputTypes: readonly IRType[];
  body: Region;
  attributes: Map<string, AttrValue>;
  _version: number;

  constructor(name: string, inputTypes: readonly IRType[], outputTypes: readonly IRType[]) {
    this.name = name;
    this.inputTypes = Object.freeze([...inputTypes]);
    this.outputTypes = Object.freeze([...outputTypes]);
    this.body = new Region();
    this.attributes = new Map();
    const entryBlock = new Block(inputTypes);
    entryBlock._parentFunction = this;
    this.body.addBlock(entryBlock);
    this._version = 0;
  }

  getAttr<T extends AttrValue = AttrValue>(name: string): T | undefined {
    return this.attributes.get(name) as T | undefined;
  }

  hasAttr(name: string): boolean { return this.attributes.has(name); }

  setAttr(name: string, value: AttrValue): this {
    this.attributes.set(name, value);
    return this;
  }

  get entryBlock(): Block { return this.body.entryBlock as Block; }
  get args(): BlockArgument[] { return this.entryBlock.arguments; }

  bumpVersion(): void { this._version++; }
  get version(): number { return this._version; }

  *ops(): Generator<Operation, void, undefined> {
    for (const block of this.body) {
      yield* block;
    }
  }

  *opsRecursive(): Generator<Operation, void, undefined> {
    for (const op of this.ops()) {
      yield op;
      yield* opsInRegions(op);
    }
  }

  *blocksRecursive(): Generator<Block, void, undefined> {
    for (const block of this.body) {
      yield block;
      for (const op of block) yield* blocksInRegions(op);
    }
  }

  opsArray(): Operation[] {
    const result: Operation[] = [];
    for (const block of this.body) {
      for (const op of block) {
        result.push(op);
      }
    }
    return result;
  }

  numOps(): number {
    let count = 0;
    for (const block of this.body) {
      count += block.size;
    }
    return count;
  }

  findOp(predicate: (op: Operation) => boolean): Operation | null {
    for (const op of this.ops()) {
      if (predicate(op)) return op;
    }
    return null;
  }

  findOps(predicate: (op: Operation) => boolean): Operation[] {
    const result: Operation[] = [];
    for (const op of this.ops()) {
      if (predicate(op)) result.push(op);
    }
    return result;
  }

  getReturnOp(): Operation | null {
    const last = this.entryBlock.lastOp;
    if (last && last.opName === 'return') return last;
    return null;
  }

  getReturnValues(): Value[] {
    const ret = this.getReturnOp();
    return ret ? [...ret.operands] : [];
  }

  verify(): string[] {
    const errors: string[] = [];
    if (!this.entryBlock) {
      errors.push('Function has no entry block');
      return errors;
    }
    if (this.entryBlock.arguments.length !== this.inputTypes.length) {
      errors.push(`Entry block has ${this.entryBlock.arguments.length} args but function expects ${this.inputTypes.length}`);
    }
    const ret = this.getReturnOp();
    if (!ret) {
      errors.push('Function body has no return op');
    } else if (ret.numOperands !== this.outputTypes.length) {
      errors.push(`Return has ${ret.numOperands} operands but function declares ${this.outputTypes.length} outputs`);
    }
    return errors;
  }
}

function topoOrderTopLevel(block: Block): Operation[] {
  const arr = block.opsArray();
  const inBlock = new Set(arr);
  return topoSortByOperands(arr, (op) => inBlock.has(op), 'ignore');
}

export function cloneGraphFunction(func: GraphFunction): GraphFunction {
  const clone = new GraphFunction(func.name, func.inputTypes, func.outputTypes);
  clone.attributes = new Map(func.attributes);
  const valueMap = new Map<Value, Value>();
  const srcBlock = func.entryBlock;
  const dstBlock = clone.entryBlock;

  for (let i = 0; i < srcBlock.arguments.length; i++) {
    valueMap.set(srcBlock.arguments[i], dstBlock.arguments[i]);
  }

  const clonedByOrig = new Map<Operation, Operation>();
  for (const op of topoOrderTopLevel(srcBlock)) {
    clonedByOrig.set(op, op.clone(valueMap));
  }
  for (const op of srcBlock) {
    dstBlock.pushOp(clonedByOrig.get(op) as Operation);
  }

  clone._version = func._version;
  return clone;
}
