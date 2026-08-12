import { registry } from '../ir/graph/ops.js';
import { SideEffectKind } from '../ir/graph/op_registry.js';
import type { SideEffectMask } from '../ir/graph/op_registry.js';
import type { GraphFunction } from '../ir/graph/function.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { AnalysisCtor } from './analysis_manager.js';

export type ValueEffectEntry = { op: Operation; effect: MemoryEffect };

export class MemoryEffect {
  kind: SideEffectMask;
  value: Value;

  constructor(kind: SideEffectMask, value: Value) {
    this.kind = kind;
    this.value = value;
  }
}

export class MemoryEffectResult {
  opEffects: Map<Operation, MemoryEffect[]>;
  private _valueEffects: Map<Value, ValueEffectEntry[]>;

  constructor(opEffects: Map<Operation, MemoryEffect[]>, valueEffects: Map<Value, ValueEffectEntry[]>) {
    this.opEffects = opEffects;
    this._valueEffects = valueEffects;
  }

  hasSideEffect(op: Operation): boolean {
    const effects = this.opEffects.get(op);
    return !!effects && effects.length > 0;
  }

  getEffects(op: Operation): MemoryEffect[] {
    return this.opEffects.get(op) || [];
  }

  getEffectsOn(value: Value): ValueEffectEntry[] {
    return this._valueEffects.get(value) || [];
  }

  getReadersOf(value: Value): Operation[] {
    const list = this._valueEffects.get(value);
    if (!list) return [];
    const result: Operation[] = [];
    for (let i = 0; i < list.length; i++) {
      if (list[i].effect.kind === SideEffectKind.READ) result.push(list[i].op);
    }
    return result;
  }

  getWritersOf(value: Value): Operation[] {
    const list = this._valueEffects.get(value);
    if (!list) return [];
    const result: Operation[] = [];
    for (let i = 0; i < list.length; i++) {
      if (list[i].effect.kind === SideEffectKind.WRITE) result.push(list[i].op);
    }
    return result;
  }
}

export class MemoryEffectAnalysis {
  static get name(): string { return 'memory_effect'; }
  static get depKey(): string { return 'memoryEffect'; }
  static get dependencies(): readonly AnalysisCtor[] { return []; }

  static compute(func: GraphFunction): MemoryEffectResult {
    const opEffects = new Map<Operation, MemoryEffect[]>();
    const valueEffects = new Map<Value, ValueEffectEntry[]>();

    const addValueEffect = (value: Value, op: Operation, effect: MemoryEffect): void => {
      let list = valueEffects.get(value);
      if (!list) {
        list = [];
        valueEffects.set(value, list);
      }
      list.push({ op, effect });
    };

    const allOps = typeof func.opsRecursive === 'function' ? func.opsRecursive() : func.ops();
    for (const op of allOps) {
      const def = registry.get(op.opName);
      const effects: MemoryEffect[] = [];

      let effectKind: SideEffectMask = SideEffectKind.NONE;
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
          if (effectKind & SideEffectKind.READ) {
            for (let i = 0; i < op.numOperands; i++) {
              const eff = new MemoryEffect(SideEffectKind.READ, op.getOperand(i));
              effects.push(eff);
              addValueEffect(op.getOperand(i), op, eff);
            }
          }
          if (effectKind & SideEffectKind.WRITE) {
            for (let i = 0; i < op.numResults; i++) {
              const eff = new MemoryEffect(SideEffectKind.WRITE, op.getResult(i));
              effects.push(eff);
              addValueEffect(op.getResult(i), op, eff);
            }
          }
        }
      }

      opEffects.set(op, effects);
    }

    return new MemoryEffectResult(opEffects, valueEffects);
  }
}
