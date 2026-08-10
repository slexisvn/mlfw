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
  OUT_EWISE_FUSABLE: 'out_ewise_fusable',
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
    this.regionSpecs = config.regions || null;
    this.genericAttrs = new Map(Object.entries(config.opAttrs || {}));
  }

  setAttr(key, value) { this.genericAttrs.set(key, value); return this; }
  getAttr(key) { return this.genericAttrs.has(key) ? this.genericAttrs.get(key) : null; }
  hasAttr(key) { return this.genericAttrs.has(key); }

  hasTrait(trait) { return this.traits.has(trait); }
  addTrait(trait) { this.traits.add(trait); return this; }
  get isCommutative() { return this.traits.has(OpTrait.COMMUTATIVE); }
  get isAssociative() { return this.traits.has(OpTrait.ASSOCIATIVE); }
  get isElementwise() { return this.traits.has(OpTrait.ELEMENTWISE); }
  get isTerminator() { return this.traits.has(OpTrait.TERMINATOR); }
  get isConstant() { return this.traits.has(OpTrait.CONSTANT); }
  get isReduction() { return this.traits.has(OpTrait.REDUCTION); }
  get isBroadcast() { return this.traits.has(OpTrait.BROADCAST); }
  get isInjective() { return this.traits.has(OpTrait.INJECTIVE); }
  get isOutEWiseFusable() { return this.traits.has(OpTrait.OUT_EWISE_FUSABLE); }
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

  unregister(name) {
    return this._defs.delete(name);
  }

  registerOpAttr(opName, key, value) {
    const def = this._defs.get(opName);
    if (!def) throw new Error(`registerOpAttr: op '${opName}' not registered`);
    def.setAttr(key, value);
    return def;
  }

  registerTrait(opName, trait) {
    const def = this._defs.get(opName);
    if (!def) throw new Error(`registerTrait: op '${opName}' not registered`);
    def.addTrait(trait);
    return def;
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
