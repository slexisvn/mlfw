import { performance } from 'node:perf_hooks';
import { GraphModule } from '../ir/graph/module.js';
import { PassManager } from '../passes/pass_manager.js';
import { CanonicalizePass } from '../passes/canonicalize/canonicalize.js';
import { AlgebraicSimplificationPass } from '../passes/simplify/algebraic.js';
import { ConstantFoldPass } from '../passes/simplify/constant_fold.js';
import { CSEPass } from '../passes/simplify/cse.js';
import { DCEPass } from '../passes/simplify/dce.js';
import { FusionPass } from '../passes/fusion/fusion_pass.js';
import { EpilogueFusionPass } from '../passes/fusion/epilogue_fusion.js';
import { FusionMergerPass } from '../passes/fusion/fusion_merger.js';
import { MultiOutputFusionPass } from '../passes/fusion/multi_output_fusion.js';
import { DominatorFusionPass } from '../passes/fusion/dominator_fusion.js';
import { LayoutTransformPass } from '../passes/layout/layout_transform.js';
import { QuantizationPass } from '../passes/quantization/quantization_pass.js';
import { lowerGraphToPrimFunc } from '../passes/lowering/graph_to_tensor.js';
import { Schedule } from '../schedule/schedule.js';
import { SchedulePolicy } from '../schedule/rules.js';
import { MemoryPlanner } from '../passes/memory/memory_planning.js';
import { BackendPipeline } from '../../backend/pipeline.js';
import { RuntimeModule } from '../runtime/runtime.js';
import { Autotuner } from '../autotune/autotuner.js';
import { TensorVerifier } from '../ir/tensor/verifier.js';
import { verifyModule, verifyFunction } from '../ir/graph/verifier.js';
import { CalibrationCollector } from '../analysis/calibration.js';
import { DecompositionPass } from '../passes/decompose/decomposition_pass.js';
import { RematerializationPass } from '../passes/memory/rematerialization.js';
import { GraphPartitionPass, PartitionMaterializationPass } from '../passes/partition/partition_pass.js';
import { TargetFeatures } from '../../backend/target.js';
import { TraceLog, TraceLevel, CompilationError } from './trace.js';
import { IRPrinter } from '../ir/graph/printer.js';
import { printTensorIR } from '../ir/tensor/printer.js';
import { lowerToLIR } from '../passes/lowering/tensor_to_lir.js';

function spread(obj) { return obj && typeof obj === 'object' ? obj : {}; }

function omit(obj, ...keys) {
  const s = new Set(keys);
  const r = {};
  for (const k of Object.keys(obj)) { if (!s.has(k)) r[k] = obj[k]; }
  return r;
}

export { CompilationError } from './trace.js';

export class CompilerConfig {
  constructor(opts = {}) {
    this.target = opts.target;
    this.verify = opts.verify !== false;
    this.errorMode = opts.errorMode || 'strict';

    const f = opts.fusion || {};
    this.fusion = {
      enabled:   f.enabled   ?? opts.enableFusion ?? true,
      strategy:  f.strategy  ?? opts.fusionStrategy ?? 'xla',
      epilogue:  f.epilogue  ?? opts.enableEpilogueFusion,
      ...spread(opts.fusionConfig), ...omit(f, 'enabled', 'strategy', 'epilogue'),
    };

    const s = opts.scheduling || {};
    this.scheduling = {
      enabled:  s.enabled  ?? opts.enableSchedule ?? false,
      autotune: s.autotune ?? opts.enableAutotune ?? false,
      ...spread(opts.autotuneConfig), ...omit(s, 'enabled', 'autotune'),
    };

    const q = opts.quantization || {};
    this.quantization = {
      enabled: q.enabled ?? opts.enableQuantization ?? false,
      ...spread(opts.quantizationConfig), ...omit(q, 'enabled'),
    };

    const o = opts.optimization || {};
    this.optimization = {
      layout:            o.layout            ?? opts.enableLayoutOptimization ?? false,
      rematerialization: o.rematerialization ?? opts.enableRematerialization ?? false,
      rematConfig:       o.rematConfig       ?? opts.rematerializationConfig ?? {},
    };

    const m = opts.memory || {};
    this.memory = {
      alignment:    m.alignment    ?? opts.memoryAlignment ?? 64,
      inplaceReuse: m.inplaceReuse ?? opts.enableInplaceReuse ?? true,
    };

    const p = opts.partition || {};
    this.partition = {
      enabled: p.enabled ?? false,
      targets: p.targets || [],
      defaultTarget: p.defaultTarget || null,
      opTargetOverrides: p.opTargetOverrides || new Map(),
      memoryLimits: p.memoryLimits || new Map(),
      minPartitionSize: p.minPartitionSize || 1,
      costWeights: p.costWeights || {},
    };

    const t = opts.trace || {};
    this.trace = {
      level: t.level ?? TraceLevel.SILENT,
      sink:  t.sink  ?? null,
      irSnapshot: {
        afterGraphPasses: false,
        afterLowering: false,
        afterScheduling: false,
        ...spread(t.irSnapshot),
      },
    };
  }
}

export class CompilationResult {
  constructor(runtimeModule, trace, errors) {
    this.module = runtimeModule;
    this.trace = trace;
    this.errors = errors || [];
  }

  get succeeded() { return this.errors.length === 0; }

  get failedFunctions() {
    const names = new Set();
    for (const e of this.errors) if (e.funcName) names.add(e.funcName);
    return names;
  }

  run(funcName, ...args) {
    return this.module.run(funcName, ...args);
  }

  async runAsync(funcName, ...args) {
    return this.module.runAsync(funcName, ...args);
  }

  isAsync(funcName) {
    return this.module.isAsync(funcName);
  }

  getSource(funcName) {
    return this.module.getKernelSource(funcName);
  }

  getSnippet(funcName) {
    return this.module.getKernelSnippet(funcName);
  }

  listKernels() {
    return this.module.listKernels();
  }
}

export class Compiler {
  constructor(config) {
    this.config = config instanceof CompilerConfig ? config : new CompilerConfig(config);
    if (!this.config.target) throw new Error('Compiler requires a target');
  }

  compile(graphModule) {
    const trace = new TraceLog(this.config.trace);
    const resilient = this.config.errorMode === 'resilient';
    const errors = [];
    const failed = new Set();
    const t0 = performance.now();
    trace.phaseStart('compile');

    if (this.config.verify) {
      this._verifyGraph(graphModule, 'before graph passes', trace, errors, failed, resilient);
    }

    this._runGraphPasses(graphModule, trace, errors, failed, resilient);

    if (this.config.partition.enabled && this.config.partition.targets.length >= 2) {
      this._runPartitioning(graphModule, trace);
    }

    if (this.config.verify) {
      this._verifyGraph(graphModule, 'after graph passes', trace, errors, failed, resilient);
    }

    const primFuncs = this._lowerAll(graphModule, trace, errors, failed, resilient);

    this._scheduleAll(primFuncs, trace, errors, failed, resilient);

    this._planMemory(primFuncs, trace, errors, failed, resilient);

    if (this.config.verify) {
      this._verifyAll(primFuncs, errors, failed, resilient);
    }

    const lirFuncs = this._lowerToLIR(primFuncs, trace, errors, failed, resilient);

    const runtimeModule = this._codegen(lirFuncs, trace, errors, failed, resilient);

    trace.phaseEnd('compile', performance.now() - t0);

    if (!resilient && errors.length > 0) {
      throw new Error(errors[0].toString());
    }

    return new CompilationResult(runtimeModule, trace, errors);
  }

  compileFunction(graphFunc) {
    const mod = new GraphModule('single');
    mod.addFunction(graphFunc);
    return this.compile(mod);
  }

  calibrate(graphModule, mode = 'minmax') {
    const collector = new CalibrationCollector(mode);
    for (const func of graphModule) {
      collector.attach(func);
    }
    return collector;
  }

  _runGraphPasses(graphModule, trace, errors, failed, resilient) {
    const pm = new PassManager();

    pm.addPass(new DecompositionPass());
    pm.addPass(new CanonicalizePass());
    pm.addPass(new AlgebraicSimplificationPass());
    pm.addPass(new ConstantFoldPass());
    pm.addPass(new CSEPass());
    pm.addPass(new DCEPass());

    if (this.config.optimization.layout && this.config.target) {
      pm.addPass(new LayoutTransformPass({ target: this.config.target }));
      pm.addPass(new DCEPass());
    }

    if (this.config.quantization.enabled) {
      pm.addPass(new QuantizationPass({ ...this.config.quantization, target: this.config.target }));
      pm.addPass(new CanonicalizePass());
      pm.addPass(new DCEPass());
    }

    const shouldEpilogueFuse = this.config.fusion.epilogue !== undefined
      ? this.config.fusion.epilogue
      : (this.config.target && this.config.target.enableEpilogueFusion);

    if (shouldEpilogueFuse) {
      pm.addPass(new EpilogueFusionPass({ target: this.config.target }));
      pm.addPass(new DCEPass());
    }

    if (this.config.fusion.enabled) {
      const fCfg = this.config.fusion;
      if (fCfg.strategy === 'dominator') {
        pm.addPass(new DominatorFusionPass({ target: this.config.target, ...fCfg }));
      } else {
        pm.addPass(new FusionPass({ target: this.config.target, cost: { launchOverheadUs: 5 }, ...fCfg }));
        pm.addPass(new FusionMergerPass({ maxFusionSize: this.config.target?.maxFusionSize, ...fCfg }));
        pm.addPass(new MultiOutputFusionPass({ maxFusionSize: this.config.target?.maxFusionSize, ...fCfg }));
      }
      pm.addPass(new DCEPass());
    }

    if (this.config.optimization.rematerialization) {
      pm.addPass(new RematerializationPass(this.config.optimization.rematConfig));
    }

    pm.setTrace(trace);

    trace.phaseStart('graphPasses');
    const t0 = performance.now();
    const result = pm.run(graphModule, { errorMode: resilient ? 'resilient' : 'strict' });
    if (result.errors) {
      for (const e of result.errors) {
        errors.push(e);
        trace.errorEvent(e.phase, e.funcName, e.message, e.passName);
      }
      if (result.failedFunctions) {
        for (const name of result.failedFunctions) failed.add(name);
      }
    }
    trace.phaseEnd('graphPasses', performance.now() - t0);

    if (trace.shouldSnapshot('afterGraphPasses')) {
      const printer = new IRPrinter();
      trace.irDump('afterGraphPasses', printer.printModule(graphModule));
    }
  }

  _runPartitioning(graphModule, trace) {
    const pm = new PassManager();
    pm.addPass(new GraphPartitionPass(this.config.partition));
    pm.addPass(new PartitionMaterializationPass({
      targets: this.config.partition.targets,
    }));

    pm.setTrace(trace);

    trace.phaseStart('partition');
    const t0 = performance.now();
    pm.run(graphModule);
    trace.phaseEnd('partition', performance.now() - t0);
  }

  _lowerAll(graphModule, trace, errors, failed, resilient) {
    trace.phaseStart('lowering');
    const t0 = performance.now();
    const primFuncs = [];
    for (const func of graphModule) {
      if (failed.has(func.name)) continue;
      try {
        const ft0 = performance.now();
        const primFunc = lowerGraphToPrimFunc(func);
        trace.functionEvent('lowering', func.name, { durationMs: performance.now() - ft0 });
        primFuncs.push(primFunc);
        if (trace.shouldSnapshot('afterLowering')) {
          trace.irDump('afterLowering:' + func.name, printTensorIR(primFunc));
        }
      } catch (e) {
        const err = new CompilationError('lowering', func.name, e.message);
        errors.push(err);
        failed.add(func.name);
        trace.errorEvent('lowering', func.name, e.message);
        if (!resilient) break;
      }
    }
    trace.phaseEnd('lowering', performance.now() - t0);
    return primFuncs;
  }

  _scheduleAll(primFuncs, trace, errors, failed, resilient) {
    trace.phaseStart('scheduling');
    const t0 = performance.now();
    const sCfg = this.config.scheduling;
    if (sCfg.autotune) {
      const autotuner = new Autotuner(this.config.target, sCfg);
      for (const pf of primFuncs) {
        if (failed.has(pf.name)) continue;
        try {
          const ft0 = performance.now();
          const tuneResult = autotuner.tuneAndApply(pf);
          const durationMs = performance.now() - ft0;
          let cacheHits = 0, blockCount = 0;
          if (tuneResult && tuneResult.results) {
            blockCount = tuneResult.results.size;
            for (const [, r] of tuneResult.results) { if (r.fromCache) cacheHits++; }
          }
          trace.autotuneStats(pf.name, { durationMs, blockCount, applied: !!(tuneResult && tuneResult.applied), cacheHits });
        } catch (e) {
          const err = new CompilationError('scheduling', pf.name, e.message);
          errors.push(err);
          failed.add(pf.name);
          trace.errorEvent('scheduling', pf.name, e.message);
          if (!resilient) break;
        }
      }
    } else if (sCfg.enabled) {
      const policy = new SchedulePolicy(this.config.target);
      for (const pf of primFuncs) {
        if (failed.has(pf.name)) continue;
        try {
          const ft0 = performance.now();
          const sch = new Schedule(pf);
          policy.applyToAllBlocks(sch);
          trace.functionEvent('scheduling', pf.name, { durationMs: performance.now() - ft0 });
        } catch (e) {
          const err = new CompilationError('scheduling', pf.name, e.message);
          errors.push(err);
          failed.add(pf.name);
          trace.errorEvent('scheduling', pf.name, e.message);
          if (!resilient) break;
        }
      }
    }
    trace.phaseEnd('scheduling', performance.now() - t0);

    if (trace.shouldSnapshot('afterScheduling')) {
      for (const pf of primFuncs) {
        if (!failed.has(pf.name)) trace.irDump('afterScheduling:' + pf.name, printTensorIR(pf));
      }
    }
  }

  _planMemory(primFuncs, trace, errors, failed, resilient) {
    trace.phaseStart('memoryPlanning');
    const t0 = performance.now();
    const alignment = this.config.memory.alignment || this.config.target?.cacheLineSizeBytes || 64;
    const planner = new MemoryPlanner({ alignment, enableInplace: this.config.memory.inplaceReuse });
    for (const pf of primFuncs) {
      if (failed.has(pf.name)) continue;
      try {
        const ft0 = performance.now();
        const { plan } = planner.planAndRewrite(pf);
        const report = plan.getReport();
        trace.memoryStats(pf.name, {
          durationMs: performance.now() - ft0,
          peakMemory: report.peakMemory,
          totalTemporaries: report.totalTemporaries,
          totalInplace: report.totalInplace,
        });
      } catch (e) {
        const err = new CompilationError('memoryPlanning', pf.name, e.message);
        errors.push(err);
        failed.add(pf.name);
        trace.errorEvent('memoryPlanning', pf.name, e.message);
        if (!resilient) break;
      }
    }
    trace.phaseEnd('memoryPlanning', performance.now() - t0);
  }

  _verifyGraph(graphModule, phase, trace, errors, failed, resilient) {
    if (resilient) {
      for (const func of graphModule) {
        if (failed.has(func.name)) continue;
        const funcErrors = verifyFunction ? verifyFunction(func) : [];
        if (funcErrors.length > 0) {
          const msg = funcErrors.map(e => e.toString()).join('; ');
          errors.push(new CompilationError('verification', func.name, msg));
          failed.add(func.name);
          trace.errorEvent('verification', func.name, msg);
        }
      }
      return;
    }
    const moduleErrors = verifyModule(graphModule);
    if (moduleErrors.length > 0) {
      throw new Error('Graph verification failed (' + phase + '): ' + moduleErrors.map(e => e.toString()).join('; '));
    }
  }

  _verifyAll(primFuncs, errors, failed, resilient) {
    const verifier = new TensorVerifier();
    for (const pf of primFuncs) {
      if (failed.has(pf.name)) continue;
      const pfErrors = verifier.verify(pf);
      if (pfErrors.length > 0) {
        const msg = pfErrors.join('; ');
        if (resilient) {
          errors.push(new CompilationError('verification', pf.name, msg));
          failed.add(pf.name);
        } else {
          throw new Error('TensorIR verification failed for ' + pf.name + ': ' + msg);
        }
      }
    }
  }

  _lowerToLIR(primFuncs, trace, errors, failed, resilient) {
    trace.phaseStart('lirLowering');
    const t0 = performance.now();
    const lirFuncs = [];
    for (const pf of primFuncs) {
      if (failed.has(pf.name)) continue;
      try {
        const ft0 = performance.now();
        const lirFunc = lowerToLIR(pf, this.config.target);
        trace.functionEvent('lirLowering', pf.name, { durationMs: performance.now() - ft0 });
        lirFuncs.push(lirFunc);
      } catch (e) {
        const err = new CompilationError('lirLowering', pf.name, e.message);
        errors.push(err);
        failed.add(pf.name);
        trace.errorEvent('lirLowering', pf.name, e.message);
        if (!resilient) break;
      }
    }
    trace.phaseEnd('lirLowering', performance.now() - t0);
    return lirFuncs;
  }

  _codegen(primFuncs, trace, errors, failed, resilient) {
    trace.phaseStart('codegen');
    const t0 = performance.now();
    const runtimeMod = new RuntimeModule('compiled');

    const usePartition = this.config.partition.enabled && this.config.partition.targets.length >= 2;
    const backendCache = new Map();
    const getBackend = (target) => {
      if (!backendCache.has(target.name)) backendCache.set(target.name, new BackendPipeline(target));
      return backendCache.get(target.name);
    };
    const defaultBackend = usePartition ? null : new BackendPipeline(this.config.target);

    for (const pf of primFuncs) {
      if (failed.has(pf.name)) continue;
      try {
        const ft0 = performance.now();
        let backend;
        if (usePartition) {
          const targetName = pf._partitionTarget;
          const target = targetName
            ? this.config.partition.targets.find(t => t.name === targetName)
            : this.config.target;
          backend = getBackend(target || this.config.target);
        } else {
          backend = defaultBackend;
        }
        const compiled = backend.compile(pf);
        runtimeMod.addCompiledKernel(compiled);
        if (pf.shapeParamMap && pf.shapeParamMap.size > 0) {
          runtimeMod.setShapeParamMap(pf.name, pf.shapeParamMap, pf.bufferMap);
        }
        trace.codegenStats(pf.name, {
          durationMs: performance.now() - ft0,
          sourceSize: compiled.source.length,
          targetName: compiled.target.name,
        });
      } catch (e) {
        const err = new CompilationError('codegen', pf.name, e.message);
        errors.push(err);
        failed.add(pf.name);
        trace.errorEvent('codegen', pf.name, e.message);
        if (!resilient) break;
      }
    }

    trace.phaseEnd('codegen', performance.now() - t0);
    return runtimeMod;
  }
}

export function compileGraph(graphFunc, target, opts = {}) {
  const compiler = new Compiler({ target, ...opts });
  return compiler.compileFunction(graphFunc);
}
