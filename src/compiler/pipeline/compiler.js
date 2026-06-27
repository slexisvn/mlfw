import { GraphModule, cloneGraphModule } from '../ir/graph/module.js';
import { cloneGraphFunction } from '../ir/graph/function.js';
import { CompilerContext } from './compiler_context.js';
import { PassManager } from '../passes/pass_manager.js';
import { buildGraphPipeline } from './graph_pipeline.js';
import { lowerGraphToPrimFunc } from '../passes/lowering/graph_to_tensor.js';
import { Schedule } from '../schedule/schedule.js';
import { SchedulePolicy } from '../schedule/rules.js';
import { MemoryPlanner } from '../passes/memory/memory_planning.js';
import { BackendPipeline, detectPureMatmul } from '../../backend/pipeline.js';
import { RuntimeModule } from '../../runtime/runtime.js';
import { Autotuner } from '../autotune/autotuner.js';
import { TensorVerifier } from '../ir/tensor/verifier.js';
import { verifyModule, verifyFunction } from '../ir/graph/verifier.js';
import { CalibrationCollector } from '../analysis/calibration.js';
import { GraphPartitionPass, PartitionMaterializationPass } from '../passes/partition/partition_pass.js';
import { splitGraph } from './graph_split.js';

import { TraceLog, TraceLevel, CompilationError } from './trace.js';
import { IRPrinter } from '../ir/graph/printer.js';
import { printTensorIR } from '../ir/tensor/printer.js';
import { lowerToLIR } from '../passes/lowering/tensor_to_lir.js';
import { verifyLIR } from '../ir/lir/verifier.js';

export { CompilationError } from './trace.js';

export class CompilerConfig {
  constructor(opts = {}) {
    this.target = opts.target;
    this.verify = opts.verify !== false;
    this.verifyMode = opts.verify === 'full' ? 'full' : 'normal';
    this.errorMode = opts.errorMode || 'strict';

    const isWebGPU = this.target && typeof this.target.isWebGPU === 'function' && this.target.isWebGPU();

    this.fusion = { enabled: true, strategy: 'xla', epilogue: undefined, ...opts.fusion };
    this.scheduling = { enabled: isWebGPU, autotune: false, ...opts.scheduling };
    this.matmulBackend = opts.matmulBackend || 'native';
    this.quantization = { enabled: false, ...opts.quantization };
    this.optimization = {
      layout: false,
      rematerialization: false,
      rematConfig: {},
      fastMath: false,
      maxSimplifyIterations: 8,
      ...opts.optimization,
    };
    this.memory = {
      alignment: 64,
      inplaceReuse: true,
      allocStrategy: 'best-fit',
      poolAllocation: false,
      ...opts.memory,
    };
    this.partition = {
      enabled: false,
      targets: [],
      defaultTarget: null,
      opTargetOverrides: new Map(),
      memoryLimits: new Map(),
      minPartitionSize: 1,
      costWeights: {},
      ...opts.partition,
    };

    this.passContext = opts.passContext || null;
    this.loweringRules = opts.loweringRules || null;
    this.codegenEntries = opts.codegenEntries || null;

    const t = opts.trace || {};
    this.trace = {
      level: t.level ?? TraceLevel.SILENT,
      sink:  t.sink  ?? null,
      irSnapshot: {
        afterGraphPasses: false,
        afterLowering: false,
        afterScheduling: false,
        ...(t.irSnapshot || {}),
      },
    };
  }

  get usePartition() {
    return this.partition.enabled && this.partition.targets.length >= 2;
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
    this.context = new CompilerContext({
      loweringRules: this.config.loweringRules,
      codegenEntries: this.config.codegenEntries,
    });
  }

  compile(graphModule) {
    const trace = new TraceLog(this.config.trace);
    const resilient = this.config.errorMode === 'resilient';
    const errors = [];
    const failed = new Set();
    const t0 = performance.now();
    trace.phaseStart('compile');

    const ctx = {
      compiler: this,
      trace,
      errors,
      failed,
      resilient,
      original: graphModule,
      working: resilient ? cloneGraphModule(graphModule) : graphModule,
      cudaMatmulChain: false,
      split: null,
      primFuncs: null,
      lirFuncs: null,
      runtimeModule: null,
    };

    for (const phase of this._compilePhases()) {
      if (phase.when && !phase.when(ctx)) continue;
      phase.run(ctx);
    }

    trace.phaseEnd('compile', performance.now() - t0);

    if (!resilient && errors.length > 0) {
      throw new Error(errors[0].toString());
    }

    return new CompilationResult(ctx.runtimeModule, trace, errors);
  }

  _compilePhases() {
    return [
      {
        name: 'verify:pre',
        when: (ctx) => ctx.compiler.config.verify,
        run: (ctx) => ctx.compiler._verifyGraph(ctx.working, 'before graph passes', ctx.trace, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'graphPasses',
        run: (ctx) => { ctx.cudaMatmulChain = ctx.compiler._runGraphPasses(ctx.working, ctx.original, ctx.trace, ctx.errors, ctx.failed, ctx.resilient); },
      },
      {
        name: 'partition',
        when: (ctx) => ctx.compiler.config.usePartition,
        run: (ctx) => ctx.compiler._runPartitioning(ctx.working, ctx.trace),
      },
      {
        name: 'split',
        run: (ctx) => {
          const cfg = ctx.compiler.config;
          const isWebGPU = typeof cfg.target.isWebGPU === 'function' && cfg.target.isWebGPU();
          ctx.split = splitGraph(ctx.working, { config: cfg, target: cfg.target, cudaMatmulChain: ctx.cudaMatmulChain, isWebGPU });
        },
      },
      {
        name: 'verify:post',
        when: (ctx) => ctx.compiler.config.verify,
        run: (ctx) => ctx.compiler._verifyGraph(ctx.working, 'after graph passes', ctx.trace, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'lowering',
        run: (ctx) => {
          ctx.primFuncs = ctx.compiler._lowerAll(ctx.working, ctx.trace, ctx.errors, ctx.failed, ctx.resilient);
          if (ctx.compiler.config.matmulBackend === 'cublas') {
            for (const pf of ctx.primFuncs) {
              pf.cublasInfo = ctx.split && ctx.split.cublasInfos ? (ctx.split.cublasInfos.get(pf.name) || null) : detectPureMatmul(pf);
            }
          }
        },
      },
      {
        name: 'scheduling',
        run: (ctx) => ctx.compiler._scheduleAll(ctx.primFuncs, ctx.trace, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'verify:tensor-full',
        when: (ctx) => ctx.compiler.config.verifyMode === 'full',
        run: (ctx) => ctx.compiler._verifyAll(ctx.primFuncs, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'memoryPlanning',
        run: (ctx) => ctx.compiler._planMemory(ctx.primFuncs, ctx.trace, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'verify:tensor',
        when: (ctx) => ctx.compiler.config.verify,
        run: (ctx) => ctx.compiler._verifyAll(ctx.primFuncs, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'lirLowering',
        run: (ctx) => { ctx.lirFuncs = ctx.compiler._lowerToLIR(ctx.primFuncs, ctx.trace, ctx.errors, ctx.failed, ctx.resilient); },
      },
      {
        name: 'codegen',
        run: (ctx) => {
          ctx.runtimeModule = ctx.compiler._codegen(ctx.lirFuncs, ctx.trace, ctx.errors, ctx.failed, ctx.resilient);
          if (ctx.split) ctx.runtimeModule.executionPlan = ctx.split.plan;
        },
      },
    ];
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

  _runGraphPasses(graphModule, original, trace, errors, failed, resilient) {
    const pm = new PassManager();

    let dotCount = 0;
    for (const func of graphModule) {
      for (const op of func.ops()) {
        if (op.opName === 'dot') dotCount++;
      }
    }
    const tgt = this.config.target;
    const chainThreshold = (tgt.getAttr && tgt.getAttr('matmulChainThreshold')) ?? (tgt.kind === 'cuda' ? 2 : Infinity);
    const cudaMatmulChain = dotCount >= chainThreshold;

    for (const p of buildGraphPipeline(this.config, this.config.target, { cudaMatmulChain, context: this.context })) {
      pm.addPass(p);
    }

    pm.setTrace(trace);

    if (this.config.verifyMode === 'full') {
      pm.setVerifyHook((target, isModule) => {
        const found = isModule ? verifyModule(target) : verifyFunction(target);
        return found.map(e => e.toString());
      });
    }

    trace.phaseStart('graphPasses');
    const t0 = performance.now();
    const result = pm.run(graphModule, { errorMode: resilient ? 'resilient' : 'strict', passContext: this.config.passContext });
    if (result.errors) {
      for (const e of result.errors) {
        errors.push(e);
        trace.errorEvent(e.phase, e.funcName, e.message, e.passName);
      }
      if (result.failedFunctions) {
        for (const name of result.failedFunctions) {
          failed.add(name);
          if (resilient && original && original !== graphModule) {
            const orig = original.getFunction(name);
            if (orig) graphModule.addFunction(cloneGraphFunction(orig));
          }
        }
      }
    }
    trace.phaseEnd('graphPasses', performance.now() - t0);

    if (trace.shouldSnapshot('afterGraphPasses')) {
      const printer = new IRPrinter();
      trace.irDump('afterGraphPasses', printer.printModule(graphModule));
    }

    return cudaMatmulChain;
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
        const primFunc = lowerGraphToPrimFunc(func, this.config.target, this.context);
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
      const autotuner = new Autotuner(this.config.target, sCfg, trace);
      for (const pf of primFuncs) {
        if (failed.has(pf.name)) continue;
        if (pf.cublasInfo) continue;
        try {
          const ft0 = performance.now();
          const tuneResult = autotuner.tuneAndApply(pf);
          const durationMs = performance.now() - ft0;
          let cacheHits = 0, blockCount = 0;
          if (tuneResult && tuneResult.results) {
            blockCount = tuneResult.results.size;
            for (const [blockName, r] of tuneResult.results) {
              if (r.fromCache) cacheHits++;
              if (trace.explainsEnabled) {
                trace.explain('schedule', blockName, r.sketchName,
                  `autotuned: best of search${r.fromCache ? ' (cached)' : ''}, score ${r.score != null ? r.score.toFixed(3) : 'n/a'}`,
                  { target: this.config.target.name, params: r.params });
              }
            }
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
      const policy = new SchedulePolicy(this.config.target, null, trace);
      for (const pf of primFuncs) {
        if (failed.has(pf.name)) continue;
        if (pf.cublasInfo) continue;
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
    const planner = new MemoryPlanner({ alignment, enableInplace: this.config.memory.inplaceReuse, allocStrategy: this.config.memory.allocStrategy, poolAllocation: this.config.memory.poolAllocation });
    for (const pf of primFuncs) {
      if (failed.has(pf.name)) continue;
      if (pf.gpuRegisterBlocked) continue;
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
        if (pf.cublasInfo) lirFunc.cublasInfo = pf.cublasInfo;
        if (pf.gpuRegisterBlocked) lirFunc.gpuRegisterBlocked = true;
        if (this.config.verifyMode === 'full') {
          const lirErrors = verifyLIR(lirFunc);
          if (lirErrors.length > 0) {
            throw new Error('LIR verification failed: ' + lirErrors.map(e => e.toString()).join('; '));
          }
        }
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

    const usePartition = this.config.usePartition;
    const backendOpts = { matmulBackend: this.config.matmulBackend, context: this.context };
    const backendCache = new Map();
    const getBackend = (target) => {
      if (!backendCache.has(target.name)) backendCache.set(target.name, new BackendPipeline(target, backendOpts));
      return backendCache.get(target.name);
    };
    const defaultBackend = usePartition ? null : new BackendPipeline(this.config.target, backendOpts);

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
