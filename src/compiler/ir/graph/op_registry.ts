import type { AttrValue, Dim, IRType } from './types.js';
import type { Operation } from './operation.js';
import type { Value } from './value.js';
import type { Pattern } from '../rewrite/pattern.js';
import type { MemoryEffect } from '../../analysis/memory_effect.js';

export const SideEffectKind = Object.freeze({
  NONE: 0,
  READ: 1,
  WRITE: 2,
  ALLOCATE: 4,
  CONTROL: 8
});

export const OpAttrKey = Object.freeze({
  GPU_CAPABLE: 'gpuCapable',
  LAUNCH_BOUNDARY: 'launchBoundary',
  SEQUENTIAL_REGION: 'sequentialRegion',
  INFER_LAYOUT: 'inferLayout',
  LAYOUT_SENSITIVITY: 'layoutSensitivity',
  UNIFIED_OPERANDS: 'unifiedOperands',
});

export const OpTrait = Object.freeze({
  COMMUTATIVE: 'commutative',
  ASSOCIATIVE: 'associative',
  IDEMPOTENT: 'idempotent',
  ELEMENTWISE: 'elementwise',
  SAME_OPERAND_AND_RESULT_TYPE: 'same_type',
  SAME_OPERAND_AND_RESULT_SHAPE: 'same_shape',
  TERMINATOR: 'terminator',
  CONSTANT: 'constant',
  BROADCAST: 'broadcast',
  REDUCTION: 'reduction',
  VIEW: 'view',
  INJECTIVE: 'injective',
  OUT_EWISE_FUSABLE: 'out_ewise_fusable',
  OPAQUE: 'opaque'
});

export type SideEffectMask = number;
export type OpTraitValue = (typeof OpTrait)[keyof typeof OpTrait];

export type AttrSpecType = 'any' | 'array' | 'boolean' | 'number' | 'object' | 'string';

export type AttrSpec = Readonly<{
  name: string;
  type: AttrSpecType;
  required?: boolean;
}>;

export type RegionSpec = Readonly<{
  name: string;
  numBlocks?: number;
}>;

export type OpAttrRecord = Readonly<Record<string, AttrValue>>;
export type OpAttrMap = ReadonlyMap<string, AttrValue>;

export type InferResultTypesFn = (
  operandTypes: readonly IRType[],
  attrs: OpAttrMap,
  explicitResultTypes: readonly IRType[] | null,
) => readonly IRType[] | null;

export type SymbolicDim = Dim | null;
export type PropagatedShape = readonly SymbolicDim[];

export type PropagateSymbolicShapesFn = (
  op: Operation,
  shapes: ReadonlyMap<Value, PropagatedShape>,
) => PropagatedShape[] | null;

export type VerifyFn = (op: Operation) => string[];

export type MemoryEffectsFn = (op: Operation) => MemoryEffect[];

export type FoldFn = (
  constValues: readonly AttrValue[],
  attrs: OpAttrMap,
  constOps: readonly Operation[],
) => AttrValue | undefined;

export type CanonicalizationPatternsFn = () => Pattern[];

export type FlopsFn = (op: Operation) => number;

export type OpDefConfig = Readonly<{
  name: string;
  numOperands: number;
  numResults?: number;
  attrs?: readonly AttrSpec[];
  sideEffects?: SideEffectMask;
  traits?: readonly OpTraitValue[];
  inferResultTypes?: InferResultTypesFn;
  propagateSymbolicShapes?: PropagateSymbolicShapesFn;
  verify?: VerifyFn;
  getMemoryEffects?: MemoryEffectsFn;
  fold?: FoldFn;
  getCanonicalizationPatterns?: CanonicalizationPatternsFn;
  getFlops?: FlopsFn;
  hasRegions?: boolean;
  numRegions?: number;
  regions?: readonly RegionSpec[];
  opAttrs?: Readonly<Record<string, unknown>>;
}>;

export class OpDef {
  name: string;
  numOperands: number;
  numResults: number;
  attrs: readonly AttrSpec[];
  sideEffects: SideEffectMask;
  traits: Set<OpTraitValue>;
  inferResultTypes: InferResultTypesFn | null;
  propagateSymbolicShapes: PropagateSymbolicShapesFn | null;
  verify: VerifyFn | null;
  getMemoryEffects: MemoryEffectsFn | null;
  fold: FoldFn | null;
  getCanonicalizationPatterns: CanonicalizationPatternsFn | null;
  getFlops: FlopsFn | null;
  hasRegions: boolean;
  numRegions: number;
  regionSpecs: readonly RegionSpec[] | null;
  genericAttrs: Map<string, unknown>;

  constructor(config: OpDefConfig) {
    this.name = config.name;
    this.numOperands = config.numOperands;
    this.numResults = config.numResults !== undefined ? config.numResults : 1;
    this.attrs = Object.freeze(config.attrs || []);
    this.sideEffects = config.sideEffects || SideEffectKind.NONE;
    this.traits = new Set(config.traits || []);
    this.inferResultTypes = config.inferResultTypes || null;
    this.propagateSymbolicShapes = config.propagateSymbolicShapes || null;
    this.verify = config.verify || null;
    this.getMemoryEffects = config.getMemoryEffects || null;
    this.fold = config.fold || null;
    this.getCanonicalizationPatterns = config.getCanonicalizationPatterns || null;
    this.getFlops = config.getFlops || null;
    this.hasRegions = config.hasRegions || false;
    this.numRegions = config.numRegions || 0;
    this.regionSpecs = config.regions || null;
    this.genericAttrs = new Map(Object.entries(config.opAttrs || {}));
  }

  setAttr(key: string, value: unknown): this { this.genericAttrs.set(key, value); return this; }
  getAttr<T = unknown>(key: string): T | null { return this.genericAttrs.has(key) ? this.genericAttrs.get(key) as T : null; }
  hasAttr(key: string): boolean { return this.genericAttrs.has(key); }

  hasTrait(trait: OpTraitValue): boolean { return this.traits.has(trait); }
  addTrait(trait: OpTraitValue): this { this.traits.add(trait); return this; }
  get isCommutative(): boolean { return this.traits.has(OpTrait.COMMUTATIVE); }
  get isAssociative(): boolean { return this.traits.has(OpTrait.ASSOCIATIVE); }
  get isElementwise(): boolean { return this.traits.has(OpTrait.ELEMENTWISE); }
  get isTerminator(): boolean { return this.traits.has(OpTrait.TERMINATOR); }
  get isConstant(): boolean { return this.traits.has(OpTrait.CONSTANT); }
  get isReduction(): boolean { return this.traits.has(OpTrait.REDUCTION); }
  get isBroadcast(): boolean { return this.traits.has(OpTrait.BROADCAST); }
  get isInjective(): boolean { return this.traits.has(OpTrait.INJECTIVE); }
  get isView(): boolean { return this.traits.has(OpTrait.VIEW); }
  get isOutEWiseFusable(): boolean { return this.traits.has(OpTrait.OUT_EWISE_FUSABLE); }
  get isOpaque(): boolean { return this.traits.has(OpTrait.OPAQUE); }
  get hasSideEffects(): boolean { return this.sideEffects !== SideEffectKind.NONE; }
}

export class OpRegistry {
  private _defs: Map<string, OpDef>;

  constructor() {
    this._defs = new Map();
  }

  register(opDef: OpDef): void {
    if (this._defs.has(opDef.name)) {
      throw new Error(`Op '${opDef.name}' already registered`);
    }
    this._defs.set(opDef.name, opDef);
  }

  unregister(name: string): boolean {
    return this._defs.delete(name);
  }

  registerOpAttr(opName: string, key: string, value: unknown): OpDef {
    const def = this._defs.get(opName);
    if (!def) throw new Error(`registerOpAttr: op '${opName}' not registered`);
    def.setAttr(key, value);
    return def;
  }

  registerTrait(opName: string, trait: OpTraitValue): OpDef {
    const def = this._defs.get(opName);
    if (!def) throw new Error(`registerTrait: op '${opName}' not registered`);
    def.addTrait(trait);
    return def;
  }

  get(name: string): OpDef | null {
    return this._defs.get(name) || null;
  }

  has(name: string): boolean {
    return this._defs.has(name);
  }

  allOps(): OpDef[] {
    return [...this._defs.values()];
  }

  names(): string[] {
    return [...this._defs.keys()];
  }
}
