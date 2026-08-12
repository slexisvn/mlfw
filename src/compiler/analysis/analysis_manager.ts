import type { GraphFunction } from '../ir/graph/function.js';

export type AnalysisDeps = Record<string, unknown>;

export type AnalysisCtor<TResult = unknown> = {
  readonly name: string;
  readonly depKey?: string;
  readonly dependencies?: readonly AnalysisCtor[];
  compute(func: GraphFunction, deps: AnalysisDeps): TResult;
};

type CacheEntry = { data: unknown; version: number };
type FuncCache = Map<AnalysisCtor, CacheEntry>;

export class AnalysisManager {
  private _cache: WeakMap<GraphFunction, FuncCache>;

  constructor() {
    this._cache = new WeakMap();
  }

  getAnalysis<TResult>(AnalysisClass: AnalysisCtor<TResult>, func: GraphFunction): TResult {
    let funcCache = this._cache.get(func);
    if (!funcCache) {
      funcCache = new Map();
      this._cache.set(func, funcCache);
    }

    let result = funcCache.get(AnalysisClass as AnalysisCtor);
    if (!result || result.version !== func.version) {
      const deps = this._resolveDeps(AnalysisClass as AnalysisCtor, func, funcCache);
      const data = AnalysisClass.compute(func, deps);
      result = { data, version: func.version };
      funcCache.set(AnalysisClass as AnalysisCtor, result);
    }

    return result.data as TResult;
  }

  _resolveDeps(AnalysisClass: AnalysisCtor, func: GraphFunction, funcCache: FuncCache): AnalysisDeps {
    const depClasses = AnalysisClass.dependencies;
    if (!depClasses || depClasses.length === 0) return {};
    const deps: AnalysisDeps = {};
    for (const Dep of depClasses) {
      const key = Dep.depKey || Dep.name;
      deps[key] = this.getAnalysis(Dep, func);
    }
    return deps;
  }

  invalidate(func: GraphFunction, preservedSet: ReadonlySet<AnalysisCtor | string> | null = null): void {
    const funcCache = this._cache.get(func);
    if (!funcCache) return;

    if (!preservedSet) {
      this._cache.delete(func);
      return;
    }

    const isPreserved = (cls: AnalysisCtor): boolean => preservedSet.has(cls) || preservedSet.has(cls.name);

    const staleMemo = new Map<AnalysisCtor, boolean>();
    const isStale = (cls: AnalysisCtor): boolean => {
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

    const toDelete: AnalysisCtor[] = [];
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

  invalidateFunctions(funcs: Iterable<GraphFunction>, preservedSet: ReadonlySet<AnalysisCtor | string> | null = null): void {
    for (const func of funcs) this.invalidate(func, preservedSet);
  }

  invalidateAll(): void {
    this._cache = new WeakMap();
  }
}
