export const SideEffectKind = Object.freeze({
  NONE: 0,
  READ: 1,
  WRITE: 2,
  ALLOCATE: 4,
  CONTROL: 8
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
  OPAQUE: 'opaque'
});

export class OpDef {
  constructor(config) {
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
  }

  hasTrait(trait) { return this.traits.has(trait); }
  get isCommutative() { return this.traits.has(OpTrait.COMMUTATIVE); }
  get isAssociative() { return this.traits.has(OpTrait.ASSOCIATIVE); }
  get isElementwise() { return this.traits.has(OpTrait.ELEMENTWISE); }
  get isTerminator() { return this.traits.has(OpTrait.TERMINATOR); }
  get isConstant() { return this.traits.has(OpTrait.CONSTANT); }
  get isReduction() { return this.traits.has(OpTrait.REDUCTION); }
  get isBroadcast() { return this.traits.has(OpTrait.BROADCAST); }
  get isInjective() { return this.traits.has(OpTrait.INJECTIVE); }
  get isOpaque() { return this.traits.has(OpTrait.OPAQUE); }
  get hasSideEffects() { return this.sideEffects !== SideEffectKind.NONE; }
}

export class OpRegistry {
  constructor() {
    this._defs = new Map();
  }

  register(opDef) {
    if (this._defs.has(opDef.name)) {
      throw new Error(`Op '${opDef.name}' already registered`);
    }
    this._defs.set(opDef.name, opDef);
  }

  get(name) {
    return this._defs.get(name) || null;
  }

  has(name) {
    return this._defs.has(name);
  }

  allOps() {
    return [...this._defs.values()];
  }

  names() {
    return [...this._defs.keys()];
  }
}
