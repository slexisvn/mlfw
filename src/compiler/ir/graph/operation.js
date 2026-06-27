import { Value, UseLink } from './value.js';
import { Region } from './block.js';
import { registry } from './ops.js';

let _opIdCounter = 0;

export function resetOpCounter() {
  _opIdCounter = 0;
}

export class Operation {
  constructor(opName, operands, resultTypes, attributes = null, regions = null) {
    this.opName = opName;
    this.id = _opIdCounter++;
    this.parentBlock = null;
    this._prev = null;
    this._next = null;

    this.attributes = new Map();
    if (attributes) {
      if (attributes instanceof Map) {
        for (const [k, v] of attributes) this.attributes.set(k, v);
      } else {
        for (const k of Object.keys(attributes)) this.attributes.set(k, attributes[k]);
      }
    }

    this.operands = new Array(operands.length);
    this._operandLinks = new Array(operands.length);
    for (let i = 0; i < operands.length; i++) {
      this.operands[i] = operands[i];
      const link = new UseLink(this, i);
      operands[i].addUse(link);
      this._operandLinks[i] = link;
    }

    this.results = new Array(resultTypes.length);
    for (let i = 0; i < resultTypes.length; i++) {
      this.results[i] = new Value(resultTypes[i], this, i);
    }

    this.regions = [];
    if (regions) {
      for (let i = 0; i < regions.length; i++) {
        const r = regions[i] instanceof Region ? regions[i] : new Region();
        r.parentOp = this;
        this.regions.push(r);
      }
    }
  }

  get numOperands() { return this.operands.length; }
  get numResults() { return this.results.length; }
  get numRegions() { return this.regions.length; }

  getOperand(i) { return this.operands[i]; }
  getResult(i) { return this.results[i]; }
  getRegion(i) { return this.regions[i]; }

  getAttr(name) { return this.attributes.get(name); }
  hasAttr(name) { return this.attributes.has(name); }
  setAttr(name, value) { this.attributes.set(name, value); }
  removeAttr(name) { return this.attributes.delete(name); }

  replaceOperand(index, newValue) {
    if (index < 0 || index >= this.operands.length) {
      throw new Error(`replaceOperand: index ${index} out of range for '${this.opName}' (${this.operands.length} operands)`);
    }
    if (!(newValue instanceof Value)) {
      throw new Error(`replaceOperand: new operand for '${this.opName}' is not a Value`);
    }
    const oldValue = this.operands[index];
    if (oldValue === newValue) return;
    oldValue.removeUse(this._operandLinks[index]);
    this.operands[index] = newValue;
    const link = new UseLink(this, index);
    newValue.addUse(link);
    this._operandLinks[index] = link;
    if (this.parentBlock) this.parentBlock._notifyMutation();
  }

  dropAllOperands() {
    const had = this.operands.length > 0;
    for (let i = 0; i < this.operands.length; i++) {
      this.operands[i].removeUse(this._operandLinks[i]);
    }
    this.operands = [];
    this._operandLinks = [];
    if (had && this.parentBlock) this.parentBlock._notifyMutation();
  }

  erase() {
    for (let i = 0; i < this.results.length; i++) {
      if (this.results[i].hasUses) {
        throw new Error(`Cannot erase ${this.opName}: result ${i} still has uses`);
      }
    }
    this.dropAllOperands();
    if (this.parentBlock) {
      this.parentBlock.removeOp(this);
    }
  }

  replaceAllResultsWith(newValues) {
    if (!newValues || newValues.length !== this.results.length) {
      throw new Error(`replaceAllResultsWith: '${this.opName}' has ${this.results.length} results, got ${newValues ? newValues.length : 0}`);
    }
    for (let i = 0; i < this.results.length; i++) {
      this.results[i].replaceAllUsesWith(newValues[i]);
    }
  }

  isTerminator() {
    const def = registry.get(this.opName);
    return !!(def && def.isTerminator);
  }

  hasSideEffects() {
    const def = registry.get(this.opName);
    return !!(def && def.hasSideEffects);
  }

  getParentFunction() {
    let block = this.parentBlock;
    while (block) {
      const parentOp = block.parentOp;
      if (!parentOp) {
        return block._parentFunction || null;
      }
      block = parentOp.parentBlock;
    }
    return null;
  }

  clone(valueMap = new Map()) {
    const mappedOperands = this.operands.map(v => valueMap.get(v) || v);
    const clonedRegions = this.regions.map(r => cloneRegion(r, valueMap));
    const op = new Operation(
      this.opName,
      mappedOperands,
      this.results.map(r => r.type),
      new Map(this.attributes),
      clonedRegions
    );
    for (let i = 0; i < this.results.length; i++) {
      valueMap.set(this.results[i], op.results[i]);
    }
    return op;
  }

  structuralHash() {
    let h = 0x811c9dc5;
    for (let i = 0; i < this.opName.length; i++) {
      h = ((h ^ this.opName.charCodeAt(i)) * 0x01000193) & 0x7fffffff;
    }
    for (let i = 0; i < this.operands.length; i++) {
      h = ((h ^ this.operands[i].id) * 0x01000193) & 0x7fffffff;
    }
    for (const [key, val] of this.attributes) {
      for (let i = 0; i < key.length; i++) {
        h = ((h ^ key.charCodeAt(i)) * 0x01000193) & 0x7fffffff;
      }
      h = ((h ^ hashAttrValue(val)) * 0x01000193) & 0x7fffffff;
    }
    return h;
  }

  structuralEquals(other) {
    if (this.opName !== other.opName) return false;
    if (this.operands.length !== other.operands.length) return false;
    if (this.results.length !== other.results.length) return false;
    if (this.attributes.size !== other.attributes.size) return false;
    for (let i = 0; i < this.operands.length; i++) {
      if (this.operands[i] !== other.operands[i]) return false;
    }
    for (const [key, val] of this.attributes) {
      if (!other.attributes.has(key)) return false;
      if (!attrValueEquals(val, other.attributes.get(key))) return false;
    }
    for (let i = 0; i < this.results.length; i++) {
      if (!this.results[i].type.equals(other.results[i].type)) return false;
    }
    return true;
  }
}

function hashAttrValue(val) {
  if (typeof val === 'number') return (val * 2654435761) & 0x7fffffff;
  if (typeof val === 'string') {
    let h = 0;
    for (let i = 0; i < val.length; i++) {
      h = ((h << 5) - h + val.charCodeAt(i)) & 0x7fffffff;
    }
    return h;
  }
  if (typeof val === 'boolean') return val ? 1 : 0;
  if (Array.isArray(val)) {
    let h = 0x9e3779b9;
    for (let i = 0; i < val.length; i++) {
      h = ((h ^ hashAttrValue(val[i])) * 0x01000193) & 0x7fffffff;
    }
    return h;
  }
  if (ArrayBuffer.isView(val) && val.buffer instanceof ArrayBuffer) {
    const bytes = new Uint8Array(val.buffer, val.byteOffset, val.byteLength);
    let h = 0x9e3779b9;
    h = ((h ^ bytes.length) * 0x01000193) & 0x7fffffff;
    const step = bytes.length > 256 ? Math.ceil(bytes.length / 256) : 1;
    for (let i = 0; i < bytes.length; i += step) {
      h = ((h ^ bytes[i]) * 0x01000193) & 0x7fffffff;
    }
    return h;
  }
  if (typeof val === 'object' && val !== null && typeof val.hash === 'function') {
    return val.hash();
  }
  return 0;
}

function attrValueEquals(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!attrValueEquals(a[i], b[i])) return false;
    }
    return true;
  }
  if (typeof a === 'object' && a !== null && typeof a.equals === 'function') {
    return a.equals(b);
  }
  return false;
}

export function cloneRegion(region, valueMap = new Map()) {
  const newRegion = new Region();
  for (const block of region.blocks) {
    const argTypes = block.arguments.map(a => a.type);
    const newBlock = new block.constructor(argTypes);
    for (let i = 0; i < block.arguments.length; i++) {
      valueMap.set(block.arguments[i], newBlock.arguments[i]);
    }
    for (const op of block) {
      newBlock.pushOp(op.clone(valueMap));
    }
    newRegion.addBlock(newBlock);
  }
  return newRegion;
}
