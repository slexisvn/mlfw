export class AnalysisManager {
  constructor() {
    this._cache = new WeakMap();
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

    const isPreserved = (cls) => preservedSet.has(cls) || preservedSet.has(cls.name);

    const staleMemo = new Map();
    const isStale = (cls) => {
      const cached = staleMemo.get(cls);
      if (cached !== undefined) return cached;
      const deps = cls.dependencies;
      let stale = false;
      if (deps) {
        for (const dep of deps) {
          if (!isPreserved(dep) || isStale(dep)) { stale = true; break; }
        }
      }
      staleMemo.set(cls, stale);
      return stale;
    };

    const toDelete = [];
    for (const AnalysisClass of funcCache.keys()) {
      if (!isPreserved(AnalysisClass) || isStale(AnalysisClass)) {
        toDelete.push(AnalysisClass);
      } else {
        const entry = funcCache.get(AnalysisClass);
        if (entry) entry.version = func.version;
      }
    }

    for (const cls of toDelete) {
      funcCache.delete(cls);
    }
  }

  invalidateFunctions(funcs, preservedSet = null) {
    for (const func of funcs) this.invalidate(func, preservedSet);
  }

  invalidateAll() {
    this._cache = new WeakMap();
  }
}
