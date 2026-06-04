export class AnalysisManager {
  constructor() {
    this._cache = new Map();
  }

  getAnalysis(AnalysisClass, func) {
    let funcCache = this._cache.get(func);
    if (!funcCache) {
      funcCache = new Map();
      this._cache.set(func, funcCache);
    }

    let result = funcCache.get(AnalysisClass);
    if (!result || result.version !== func.version) {
      const deps = this._resolveDeps(AnalysisClass, func, funcCache);
      const data = AnalysisClass.compute(func, deps);
      result = { data, version: func.version };
      funcCache.set(AnalysisClass, result);
    }

    return result.data;
  }

  _resolveDeps(AnalysisClass, func, funcCache) {
    const depClasses = AnalysisClass.dependencies;
    if (!depClasses || depClasses.length === 0) return {};
    const deps = {};
    for (const Dep of depClasses) {
      const key = Dep.depKey || Dep.name;
      deps[key] = this.getAnalysis(Dep, func);
    }
    return deps;
  }

  invalidate(func, preservedSet = null) {
    const funcCache = this._cache.get(func);
    if (!funcCache) return;

    if (!preservedSet) {
      this._cache.delete(func);
      return;
    }

    const toDelete = [];
    for (const AnalysisClass of funcCache.keys()) {
      if (!preservedSet.has(AnalysisClass) && !preservedSet.has(AnalysisClass.name)) {
        toDelete.push(AnalysisClass);
      }
    }

    for (const cls of toDelete) {
      funcCache.delete(cls);
      this._cascadeInvalidate(funcCache, cls);
    }
  }

  _cascadeInvalidate(funcCache, invalidatedClass) {
    for (const [AnalysisClass] of funcCache) {
      const deps = AnalysisClass.dependencies;
      if (deps && deps.includes(invalidatedClass)) {
        funcCache.delete(AnalysisClass);
        this._cascadeInvalidate(funcCache, AnalysisClass);
      }
    }
  }

  invalidateAll() {
    this._cache.clear();
  }
}
