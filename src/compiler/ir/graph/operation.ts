import { Value, UseLink } from './value.js';
import { Region } from './block.js';
import { registry } from './ops.js';
import { topoSortByOperands } from './graph_algorithms.js';
import type { AttrInit, AttrValue, HashableAttr, IRType } from './types.js';
import type { Location } from '../location.js';
import type { Block } from './block.js';
import type { GraphFunction } from './function.js';

let _opIdCounter = 0;

export function resetOpCounter(): void {
  _opIdCounter = 0;
}

export class Operation {
  opName: string;
  id: number;
  parentBlock: Block | null;
  _prev: Operation | null;
  _next: Operation | null;
  attributes: Map<string, AttrValue>;
  operands: Value[];
  _operandLinks: UseLink[];
  results: Value[];
  regions: Region[];
  loc: Location | null;

  constructor(
    opName: string,
    operands: readonly Value[],
    resultTypes: readonly IRType[],
    attributes: AttrInit | null = null,
    regions: readonly Region[] | null = null,
  ) {
    this.opName = opName;
    this.id = _opIdCounter++;
    this.parentBlock = null;
    this._prev = null;
    this._next = null;
    this.loc = null;

    this.attributes = new Map();
    if (attributes) {
      if (attributes instanceof Map) {
        for (const [k, v] of attributes) this.attributes.set(k, v);
      } else {
        const record = attributes as Readonly<Record<string, AttrValue>>;
        for (const k of Object.keys(record)) this.attributes.set(k, record[k]);
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

  get numOperands(): number { return this.operands.length; }
  get numResults(): number { return this.results.length; }
  get numRegions(): number { return this.regions.length; }

  getOperand(i: number): Value { return this.operands[i]; }
  getResult(i: number): Value { return this.results[i]; }
  getRegion(i: number): Region { return this.regions[i]; }

  getAttr<T extends AttrValue = AttrValue>(name: string): T | undefined { return this.attributes.get(name) as T | undefined; }
  hasAttr(name: string): boolean { return this.attributes.has(name); }
  setAttr(name: string, value: AttrValue): void {
    this.attributes.set(name, value);
    if (this.parentBlock) this.parentBlock._notifyMutation();
  }

  removeAttr(name: string): boolean {
    const removed = this.attributes.delete(name);
    if (removed && this.parentBlock) this.parentBlock._notifyMutation();
    return removed;
  }

  replaceOperand(index: number, newValue: Value): void {
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

  dropAllOperands(): void {
    const had = this.operands.length > 0;
    for (let i = 0; i < this.operands.length; i++) {
      this.operands[i].removeUse(this._operandLinks[i]);
    }
    this.operands = [];
    this._operandLinks = [];
    if (had && this.parentBlock) this.parentBlock._notifyMutation();
  }

  erase(): void {
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

  replaceAllResultsWith(newValues: readonly Value[] | null): void {
    if (!newValues || newValues.length !== this.results.length) {
      throw new Error(`replaceAllResultsWith: '${this.opName}' has ${this.results.length} results, got ${newValues ? newValues.length : 0}`);
    }
    for (let i = 0; i < this.results.length; i++) {
      this.results[i].replaceAllUsesWith(newValues[i]);
    }
  }

  isTerminator(): boolean {
    const def = registry.get(this.opName);
    return !!(def && def.isTerminator);
  }

  hasSideEffects(): boolean {
    const def = registry.get(this.opName);
    return !!(def && def.hasSideEffects);
  }

  getParentFunction(): GraphFunction | null {
    return this.parentBlock ? this.parentBlock._owningFunction() : null;
  }

  clone(valueMap: Map<Value, Value> = new Map()): Operation {
    const mappedOperands = this.operands.map(v => valueMap.get(v) || v);
    const clonedRegions = this.regions.map(r => cloneRegion(r, valueMap));
    const clonedAttrs = new Map<string, AttrValue>();
    for (const [k, v] of this.attributes) clonedAttrs.set(k, cloneAttrValue(v));
    const op = new Operation(
      this.opName,
      mappedOperands,
      this.results.map(r => r.type),
      clonedAttrs,
      clonedRegions
    );
    for (let i = 0; i < this.results.length; i++) {
      valueMap.set(this.results[i], op.results[i]);
    }
    op.loc = this.loc;
    return op;
  }

  hasInterchangeableOperands(): boolean {
    if (this.operands.length !== 2) return false;
    const def = registry.get(this.opName);
    return def !== null && def.isCommutative;
  }

  structuralHash(): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < this.opName.length; i++) {
      h = ((h ^ this.opName.charCodeAt(i)) * 0x01000193) & 0x7fffffff;
    }
    if (this.hasInterchangeableOperands()) {
      const a = this.operands[0].id, b = this.operands[1].id;
      h = ((h ^ (a < b ? a : b)) * 0x01000193) & 0x7fffffff;
      h = ((h ^ (a < b ? b : a)) * 0x01000193) & 0x7fffffff;
    } else {
      for (let i = 0; i < this.operands.length; i++) {
        h = ((h ^ this.operands[i].id) * 0x01000193) & 0x7fffffff;
      }
    }
    for (const [key, val] of this.attributes) {
      for (let i = 0; i < key.length; i++) {
        h = ((h ^ key.charCodeAt(i)) * 0x01000193) & 0x7fffffff;
      }
      h = ((h ^ hashAttrValue(val)) * 0x01000193) & 0x7fffffff;
    }
    if (this.regions.length > 0) {
      h = ((h ^ (0x9e3779b9 + this.regions.length)) * 0x01000193) & 0x7fffffff;
    }
    return h;
  }

  structuralEquals(other: Operation): boolean {
    if (this.opName !== other.opName) return false;
    if (this.operands.length !== other.operands.length) return false;
    if (this.results.length !== other.results.length) return false;
    if (this.attributes.size !== other.attributes.size) return false;
    if (this.regions.length > 0 || other.regions.length > 0) return false;
    if (this.hasInterchangeableOperands()) {
      const sameOrder = this.operands[0] === other.operands[0] && this.operands[1] === other.operands[1];
      const swapped = this.operands[0] === other.operands[1] && this.operands[1] === other.operands[0];
      if (!sameOrder && !swapped) return false;
    } else {
      for (let i = 0; i < this.operands.length; i++) {
        if (this.operands[i] !== other.operands[i]) return false;
      }
    }
    for (const [key, val] of this.attributes) {
      if (!other.attributes.has(key)) return false;
      if (!attrValueEquals(val, other.attributes.get(key) as AttrValue)) return false;
    }
    for (let i = 0; i < this.results.length; i++) {
      if (!this.results[i].type.equals(other.results[i].type)) return false;
    }
    return true;
  }
}

function cloneAttrValue(v: AttrValue): AttrValue {
  if (Array.isArray(v)) return (v as readonly AttrValue[]).map(cloneAttrValue);
  return v;
}

function hashAttrValue(val: AttrValue): number {
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
    const arr = val as readonly AttrValue[];
    let h = 0x9e3779b9;
    for (let i = 0; i < arr.length; i++) {
      h = ((h ^ hashAttrValue(arr[i])) * 0x01000193) & 0x7fffffff;
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
  if (typeof val === 'object' && val !== null && typeof (val as Partial<HashableAttr>).hash === 'function') {
    return (val as HashableAttr).hash();
  }
  return 0;
}

function attrValueEquals(a: AttrValue, b: AttrValue): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a)) {
    const av = a as readonly AttrValue[];
    if (!Array.isArray(b)) return false;
    const bv = b as readonly AttrValue[];
    if (av.length !== bv.length) return false;
    for (let i = 0; i < av.length; i++) {
      if (!attrValueEquals(av[i], bv[i])) return false;
    }
    return true;
  }
  if (typeof a === 'object' && a !== null && typeof (a as Partial<HashableAttr>).equals === 'function') {
    return (a as HashableAttr).equals(b);
  }
  return false;
}

export function cloneRegion(region: Region, valueMap: Map<Value, Value> = new Map()): Region {
  const newRegion = new Region();
  for (const block of region.blocks) {
    const argTypes = block.arguments.map(a => a.type);
    const BlockCtor = block.constructor as new (argTypes: readonly IRType[]) => Block;
    const newBlock = new BlockCtor(argTypes);
    for (let i = 0; i < block.arguments.length; i++) {
      valueMap.set(block.arguments[i], newBlock.arguments[i]);
    }
    const arr = block.opsArray();
    const inBlock = new Set(arr);
    const clonedByOrig = new Map<Operation, Operation>();
    for (const op of topoSortByOperands(arr, (o) => inBlock.has(o), 'ignore')) {
      clonedByOrig.set(op, op.clone(valueMap));
    }
    for (const op of arr) {
      newBlock.pushOp(clonedByOrig.get(op) as Operation);
    }
    newRegion.addBlock(newBlock);
  }
  return newRegion;
}
