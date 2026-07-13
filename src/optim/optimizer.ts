import type { NumericTypedArray } from '../tensor/types/dtype.js';
import type { NumberTypedArray } from './types.js';
import type {
  OptimizerParam,
  OptimizerParamGroup,
  OptimizerParams,
  OptimizerState,
  OptimizerStateDict,
  ParamGroupInput,
} from './types.js';

export class Optimizer {
  protected _defaults: Record<string, unknown>;
  protected _paramGroups: OptimizerParamGroup[];
  protected _state: Map<number, OptimizerState>;
  private readonly _paramIndex: WeakMap<OptimizerParam, number>;
  private _nextId: number;

  constructor(params: OptimizerParams, defaults: Record<string, unknown>) {
    this._defaults = { ...defaults };
    this._paramGroups = [];
    this._state = new Map();
    this._paramIndex = new WeakMap();
    this._nextId = 0;

    const groups = normalizeParamGroups(params);
    for (const group of groups) {
      this._addParamGroup(group);
    }
  }

  get paramGroups(): OptimizerParamGroup[] {
    return this._paramGroups;
  }

  get defaults(): Record<string, unknown> {
    return this._defaults;
  }

  step(): void {
    throw new Error(`${this.constructor.name}.step() not implemented`);
  }

  zeroGrad(setToNone = true): void {
    for (const group of this._paramGroups) {
      for (const p of group.params) {
        if (p.grad === null) continue;
        if (setToNone) {
          p.grad = null;
        } else {
          (p.grad._impl.storage.data! as NumberTypedArray).fill(0);
        }
      }
    }
  }

  stateDict(): OptimizerStateDict {
    const clonedState = new Map<number, OptimizerState>();
    for (const [id, s] of this._state) {
      const cloned: OptimizerState = {};
      for (const key of Object.keys(s)) {
        const val = s[key];
        cloned[key] = isTypedArray(val) ? cloneTypedArray(val) : val;
      }
      clonedState.set(id, cloned);
    }
    return {
      state: clonedState,
      paramGroups: this._paramGroups.map(g => {
        const copy: Record<string, import('./types.js').OptimizerStateValue> = {};
        for (const key of Object.keys(g)) {
          if (key === 'params') continue;
          const val = g[key];
          copy[key] = (Array.isArray(val) ? [...val] : val) as import('./types.js').OptimizerStateValue;
        }
        return copy as Record<string, import('./types.js').OptimizerStateValue>;
      }),
    };
  }

  loadStateDict(dict: OptimizerStateDict): void {
    for (let i = 0; i < this._paramGroups.length; i++) {
      const saved = dict.paramGroups[i];
      for (const key of Object.keys(saved)) {
        this._paramGroups[i][key] = Array.isArray(saved[key]) ? [...saved[key]] : saved[key];
      }
    }
    this._state = new Map();
    for (const [id, s] of dict.state) {
      const cloned: OptimizerState = {};
      for (const key of Object.keys(s)) {
        const val = s[key];
        cloned[key] = isTypedArray(val) ? cloneTypedArray(val) : val;
      }
      this._state.set(id, cloned);
    }
  }

  _addParamGroup(group: ParamGroupInput): void {
    const resolved: OptimizerParamGroup = { ...this._defaults, params: [] } as OptimizerParamGroup;
    for (const key of Object.keys(group)) {
      if (key === 'params') continue;
      resolved[key] = group[key] as import('./types.js').OptimizerStateValue;
    }
    const params = Array.isArray(group.params) ? group.params : [...group.params];
    for (const p of params) {
      if (this._paramIndex.has(p)) {
        throw new Error('Parameter appears in more than one parameter group');
      }
      this._paramIndex.set(p, this._nextId++);
    }
    resolved.params = params;
    this._paramGroups.push(resolved);
  }

  _getParamId(param: OptimizerParam): number | undefined {
    return this._paramIndex.get(param);
  }

  _getState(param: OptimizerParam): OptimizerState {
    const id = this._paramIndex.get(param)!;
    let s = this._state.get(id);
    if (!s) {
      s = {};
      this._state.set(id, s);
    }
    return s;
  }
}

function normalizeParamGroups(params: OptimizerParams): ParamGroupInput[] {
  if (params == null) throw new Error('Optimizer requires at least one parameter');
  const arr = Array.isArray(params) ? params : [...params];
  if (arr.length === 0) throw new Error('Optimizer got an empty parameter list');
  if (isParamGroupInput(arr[0])) {
    return arr as ParamGroupInput[];
  }
  return [{ params: arr as OptimizerParam[] }];
}

function isTypedArray(val: unknown): val is NumericTypedArray {
  return val instanceof Float32Array || val instanceof Float64Array
    || val instanceof Int32Array || val instanceof Int16Array
    || val instanceof Int8Array || val instanceof Uint8Array
    || val instanceof Uint16Array || val instanceof Uint32Array;
}

function cloneTypedArray(val: NumericTypedArray): NumericTypedArray {
  return val.slice() as NumericTypedArray;
}

function isParamGroupInput(value: unknown): value is ParamGroupInput {
  return typeof value === 'object' && value !== null && 'params' in value;
}
