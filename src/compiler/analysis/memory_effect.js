import { registry } from '../ir/graph/ops.js';
import { SideEffectKind, OpTrait } from '../ir/graph/op_registry.js';

export class MemoryEffect {
  constructor(kind, value) {
    this.kind = kind;
    this.value = value;
  }
}

export class MemoryEffectResult {
  constructor(opEffects, valueEffects) {
    this.opEffects = opEffects;
    this._valueEffects = valueEffects;
  }

  hasSideEffect(op) {
    const effects = this.opEffects.get(op);
    return effects && effects.length > 0;
  }

  getEffects(op) {
    return this.opEffects.get(op) || [];
  }

  getEffectsOn(value) {
    return this._valueEffects.get(value) || [];
  }

  getReadersOf(value) {
    const list = this._valueEffects.get(value);
    if (!list) return [];
    const result = [];
    for (let i = 0; i < list.length; i++) {
      if (list[i].effect.kind === SideEffectKind.READ) result.push(list[i].op);
    }
    return result;
  }

  getWritersOf(value) {
    const list = this._valueEffects.get(value);
    if (!list) return [];
    const result = [];
    for (let i = 0; i < list.length; i++) {
      if (list[i].effect.kind === SideEffectKind.WRITE) result.push(list[i].op);
    }
    return result;
  }
}

export class MemoryEffectAnalysis {
  static get name() { return 'memory_effect'; }
  static get depKey() { return 'memoryEffect'; }
  static get dependencies() { return []; }

  static compute(func) {
    const opEffects = new Map();
    const valueEffects = new Map();

    const addValueEffect = (value, op, effect) => {
      let list = valueEffects.get(value);
      if (!list) {
        list = [];
        valueEffects.set(value, list);
      }
      list.push({ op, effect });
    };

    for (const op of func.ops()) {
      const def = registry.get(op.opName);
      const effects = [];

      let effectKind = SideEffectKind.NONE;
      if (def && def.sideEffects) {
        effectKind = def.sideEffects;
      } else if (op.hasSideEffects && op.hasSideEffects()) {
        effectKind = SideEffectKind.WRITE;
      }

      if (effectKind !== SideEffectKind.NONE) {
        if (def && def.getMemoryEffects) {
          const customEffects = def.getMemoryEffects(op);
          for (const eff of customEffects) {
            effects.push(eff);
            addValueEffect(eff.value, op, eff);
          }
        } else {
          for (let i = 0; i < op.numOperands; i++) {
            const eff = new MemoryEffect(SideEffectKind.READ, op.getOperand(i));
            effects.push(eff);
            addValueEffect(op.getOperand(i), op, eff);
          }
          for (let i = 0; i < op.numResults; i++) {
            const eff = new MemoryEffect(SideEffectKind.WRITE, op.getResult(i));
            effects.push(eff);
            addValueEffect(op.getResult(i), op, eff);
          }
        }
      }

      opEffects.set(op, effects);
    }

    return new MemoryEffectResult(opEffects, valueEffects);
  }
}
