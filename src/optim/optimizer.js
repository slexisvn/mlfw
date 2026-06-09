export class Optimizer {
  constructor(params, defaults) {
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

  get paramGroups() {
    return this._paramGroups;
  }

  get defaults() {
    return this._defaults;
  }

  step() {
    throw new Error(`${this.constructor.name}.step() not implemented`);
  }

  zeroGrad(setToNone = true) {
    for (const group of this._paramGroups) {
      for (const p of group.params) {
        if (p.grad === null) continue;
        if (setToNone) {
          p.grad = null;
        } else {
          p.grad._impl.storage.data.fill(0);
        }
      }
    }
  }

  stateDict() {
    const clonedState = new Map();
    for (const [id, s] of this._state) {
      const cloned = {};
      for (const key of Object.keys(s)) {
        const val = s[key];
        cloned[key] = isTypedArray(val) ? new val.constructor(val) : val;
      }
      clonedState.set(id, cloned);
    }
    return {
      state: clonedState,
      paramGroups: this._paramGroups.map(g => {
        const copy = {};
        for (const key of Object.keys(g)) {
          if (key === 'params') continue;
          const val = g[key];
          copy[key] = Array.isArray(val) ? [...val] : val;
        }
        return copy;
      }),
    };
  }

  loadStateDict(dict) {
    for (let i = 0; i < this._paramGroups.length; i++) {
      const saved = dict.paramGroups[i];
      for (const key of Object.keys(saved)) {
        this._paramGroups[i][key] = Array.isArray(saved[key]) ? [...saved[key]] : saved[key];
      }
    }
    this._state = new Map();
    for (const [id, s] of dict.state) {
      const cloned = {};
      for (const key of Object.keys(s)) {
        const val = s[key];
        cloned[key] = isTypedArray(val) ? new val.constructor(val) : val;
      }
      this._state.set(id, cloned);
    }
  }

  _addParamGroup(group) {
    const resolved = { ...this._defaults };
    for (const key of Object.keys(group)) {
      if (key === 'params') continue;
      resolved[key] = group[key];
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

  _getParamId(param) {
    return this._paramIndex.get(param);
  }

  _getState(param) {
    const id = this._paramIndex.get(param);
    let s = this._state.get(id);
    if (!s) {
      s = {};
      this._state.set(id, s);
    }
    return s;
  }
}

function normalizeParamGroups(params) {
  if (params == null) throw new Error('Optimizer requires at least one parameter');
  const arr = Array.isArray(params) ? params : [...params];
  if (arr.length === 0) throw new Error('Optimizer got an empty parameter list');
  if (arr[0] && typeof arr[0] === 'object' && 'params' in arr[0]) {
    return arr;
  }
  return [{ params: arr }];
}

function isTypedArray(val) {
  return val instanceof Float32Array || val instanceof Float64Array
    || val instanceof Int32Array || val instanceof Int16Array
    || val instanceof Int8Array || val instanceof Uint8Array
    || val instanceof Uint16Array || val instanceof Uint32Array;
}
