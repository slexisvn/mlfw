import { Block, Region } from './block.js';

function* opsInRegions(op) {
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

function* blocksInRegions(op) {
  if (!op.regions || op.regions.length === 0) return;
  for (const region of op.regions) {
    for (const block of region.blocks) {
      yield block;
      for (const inner of block.ops()) yield* blocksInRegions(inner);
    }
  }
}

export class GraphFunction {
  constructor(name, inputTypes, outputTypes) {
    this.name = name;
    this.inputTypes = Object.freeze([...inputTypes]);
    this.outputTypes = Object.freeze([...outputTypes]);
    this.body = new Region();
    const entryBlock = new Block(inputTypes);
    entryBlock._parentFunction = this;
    this.body.addBlock(entryBlock);
    this._version = 0;
  }

  get entryBlock() { return this.body.entryBlock; }
  get args() { return this.entryBlock.arguments; }

  bumpVersion() { this._version++; }
  get version() { return this._version; }

  *ops() {
    for (const block of this.body) {
      yield* block;
    }
  }

  *opsRecursive() {
    for (const op of this.ops()) {
      yield op;
      yield* opsInRegions(op);
    }
  }

  *blocksRecursive() {
    for (const block of this.body) {
      yield block;
      for (const op of block) yield* blocksInRegions(op);
    }
  }

  opsArray() {
    const result = [];
    for (const block of this.body) {
      for (const op of block) {
        result.push(op);
      }
    }
    return result;
  }

  numOps() {
    let count = 0;
    for (const block of this.body) {
      count += block.size;
    }
    return count;
  }

  findOp(predicate) {
    for (const op of this.ops()) {
      if (predicate(op)) return op;
    }
    return null;
  }

  findOps(predicate) {
    const result = [];
    for (const op of this.ops()) {
      if (predicate(op)) result.push(op);
    }
    return result;
  }

  getReturnOp() {
    const last = this.entryBlock.lastOp;
    if (last && last.opName === 'return') return last;
    return null;
  }

  getReturnValues() {
    const ret = this.getReturnOp();
    return ret ? [...ret.operands] : [];
  }

  verify() {
    const errors = [];
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

function topoOrderTopLevel(block) {
  const inBlock = new Set(block.opsArray());
  const state = new Map();
  const ordered = [];
  for (const root of inBlock) {
    if (state.get(root) !== undefined) continue;
    const stack = [{ op: root, i: 0 }];
    state.set(root, 1);
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const op = frame.op;
      if (frame.i < op.numOperands) {
        const operand = op.getOperand(frame.i);
        frame.i++;
        const def = operand && operand.definingOp;
        if (def && inBlock.has(def) && state.get(def) === undefined) {
          state.set(def, 1);
          stack.push({ op: def, i: 0 });
        }
        continue;
      }
      state.set(op, 2);
      ordered.push(op);
      stack.pop();
    }
  }
  return ordered;
}

export function cloneGraphFunction(func) {
  const clone = new GraphFunction(func.name, func.inputTypes, func.outputTypes);
  const valueMap = new Map();
  const srcBlock = func.entryBlock;
  const dstBlock = clone.entryBlock;

  for (let i = 0; i < srcBlock.arguments.length; i++) {
    valueMap.set(srcBlock.arguments[i], dstBlock.arguments[i]);
  }

  const clonedByOrig = new Map();
  for (const op of topoOrderTopLevel(srcBlock)) {
    clonedByOrig.set(op, op.clone(valueMap));
  }
  for (const op of srcBlock) {
    dstBlock.pushOp(clonedByOrig.get(op));
  }

  clone._version = func._version;
  return clone;
}
