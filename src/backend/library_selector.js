import { libraryFunc } from './dtype_map.js';

export class LibraryCall {
  constructor(opName, libraryName, functionName, constraints = {}) {
    this.opName = opName;
    this.libraryName = libraryName;
    this.functionName = functionName;
    this.constraints = constraints;
  }
}

export class LibrarySelector {
  constructor(target) {
    this.target = target;
    this._registry = new Map();
  }

  register(opName, libraryName, constraints = {}) {
    let list = this._registry.get(opName);
    if (!list) {
      list = [];
      this._registry.set(opName, list);
    }
    list.push({ libraryName, constraints });
  }

  select(opName, shape, dtype) {
    const candidates = this._registry.get(opName);
    if (!candidates) return null;
    for (const entry of candidates) {
      if (!this._matchesConstraints(entry.constraints, shape, dtype)) continue;
      const funcName = libraryFunc(opName, entry.libraryName, dtype);
      if (!funcName) continue;
      return new LibraryCall(opName, entry.libraryName, funcName, entry.constraints);
    }
    return null;
  }

  shouldUseLibrary(opName, shape, dtype) {
    return this.select(opName, shape, dtype) !== null;
  }

  _matchesConstraints(constraints, shape, dtype) {
    if (constraints.minElements) {
      let numel = 1;
      for (let i = 0; i < shape.length; i++) numel *= shape[i];
      if (numel < constraints.minElements) return false;
    }
    if (constraints.dtypes && constraints.dtypes.indexOf(dtype) < 0) return false;
    if (constraints.maxRank && shape.length > constraints.maxRank) return false;
    return true;
  }
}

export function createCPULibrarySelector(target) {
  const selector = new LibrarySelector(target);
  selector.register('dot', 'blas', { dtypes: ['f32', 'f64'], minElements: 64 });
  return selector;
}

export function createGPULibrarySelector(target) {
  const selector = new LibrarySelector(target);
  selector.register('dot', 'cublas', { dtypes: ['f32', 'f64', 'f16'] });
  selector.register('conv', 'cudnn', { dtypes: ['f32', 'f16'] });
  return selector;
}
