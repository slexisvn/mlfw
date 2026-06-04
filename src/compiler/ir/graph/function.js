import { Block, Region } from './block.js';

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
