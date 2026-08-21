import { GraphModule, cloneGraphModule } from '../ir/graph/module.js';
import { cloneGraphFunction } from '../ir/graph/function.js';
import { CompilerContext } from './compiler_context.js';
import { PassManager } from '../passes/pass_manager.js';
import type { PartitionTarget } from '../analysis/partitioner.js';
import type { PassContext } from '../passes/pass.js';
import { TirPassManager } from '../passes/tir_pass_manager.js';
import { buildGraphPipeline } from './graph_pipeline.js';
import { buildTirPipeline } from './tir_pipeline.js';
import { buildLirPipeline } from './lir_pipeline.js';
import { LirPassManager } from '../passes/lir_pass_manager.js';
import { lowerGraphToPrimFunc } from '../passes/lowering/graph_to_tensor.js';
import { TirModule } from '../ir/tensor/module.js';
import { BackendPipeline } from '../../backend/pipeline.js';
import { activeExternalCodegenProviders } from './external_codegen.js';
import type { ExternalKernelInfo } from './external_codegen.js';
import { RuntimeModule } from '../../runtime/runtime.js';
import { CalibrationCollector } from '../analysis/calibration.js';
import type { CompileFn } from '../analysis/calibrate_exec.js';
import { GraphPartitionPass, PartitionMaterializationPass } from '../passes/partition/partition_pass.js';
import { splitGraph } from './graph_split.js';
import { splitGraphForNative } from '../passes/partition/cublas_split.js';
import { assignPlanBuffers, computePlanDonations, planMemoryReport } from '../passes/memory/plan_buffer_assignment.js';
import type { AssignablePlan } from '../passes/memory/plan_buffer_assignment.js';
import { containsSequentialRegion } from '../ir/graph/op_traits.js';
import { TargetAttr, targetAttr } from './target_attrs.js';
import type { SchedulingDefaults } from './target_attrs.js';
import { detectPureConv } from '../schedule/conv_implicit_gemm.js';

import { TraceLog, TraceLevel, CompilationError } from './trace.js';
import { IRPrinter } from '../ir/graph/printer.js';
import { printTensorIR } from '../ir/tensor/printer.js';
import { lowerToLIR } from '../passes/lowering/tensor_to_lir.js';
import { VerifyLevel, normalizeVerifyLevel, checkIRInvariants } from './invariant_check.js';
import { IRLevel } from '../ir/verify.js';
import { FuncAttr } from '../ir/func_attrs.js';

export { CompilationError } from './trace.js';
export { VerifyLevel } from './invariant_check.js';

import type { GraphFunction } from '../ir/graph/function.js';
import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { LIRFunc } from '../ir/lir/nodes.js';
import type { CompileTarget, FusionConfig, MemoryConfig, OptimizationConfig, QuantizationConfig, GraphPass, TirPass } from './pipeline_types.js';
import type { VerifyLevelValue } from './invariant_check.js';
import type { TraceLogConfig, TraceSink, IRSnapshotFlags } from './trace.js';
import type { PartitionerOpts } from '../analysis/partitioner.js';
import type { GpuLaunchDiagnosis } from '../analysis/gpu_race.js';
import type { IRLevelValue } from '../ir/verify.js';

export type SchedulingConfig = { enabled?: boolean; autotune?: boolean; gpuTiling?: boolean; [key: string]: unknown };
export type PartitionConfig = {
  enabled: boolean;
  targets: readonly PartitionTarget[];
  defaultTarget: PartitionTarget | null;
  opTargetOverrides: Map<string, PartitionTarget>;
  memoryLimits: Map<string, number>;
  minPartitionSize: number;
  costWeights: Record<string, number>;
  [key: string]: unknown;
};
export type TraceConfig = { level: number; sink: TraceSink | null; irSnapshot: IRSnapshotFlags };

export type CompilerConfigOpts = Readonly<{
  target?: CompileTarget;
  verify?: string | null;
  errorMode?: string;
  fusion?: Partial<FusionConfig>;
  scheduling?: Partial<SchedulingConfig>;
  matmulBackend?: string;
  quantization?: Partial<QuantizationConfig>;
  optimization?: Partial<OptimizationConfig>;
  memory?: Partial<MemoryConfig>;
  partition?: Partial<PartitionConfig>;
  passContext?: unknown;
  loweringRules?: ReadonlyMap<string, unknown> | Record<string, unknown> | null;
  codegenEntries?: ReadonlyMap<string, unknown> | Record<string, unknown> | null;
  trace?: Partial<TraceConfig> & TraceLogConfig;
}>;

export type CompileContext = {
  compiler: Compiler;
  trace: TraceLog;
  errors: CompilationError[];
  failed: Set<string>;
  resilient: boolean;
  original: GraphModule;
  working: GraphModule;
  split: GraphSplitResult;
  tirModule: TirModule | null;
  lirFuncs: LIRFuncLike[] | null;
  runtimeModule: RuntimeModuleLike | null;
  restartFrom: string | null;
};

const MAX_RELAUNCH_ATTEMPTS = 1;
const RACE_SPLIT_MIN_BOUNDARIES = 1;

export type LIRFuncLike = { name: string; getAttr?(key: string): unknown; shapeParamMap?: Map<string, unknown>; bufferMap?: unknown };

export type GraphSplitResult = { cublasInfos?: ReadonlyMap<string, ExternalKernelInfo>; plan?: unknown } | null;

export type CompilePhase = {
  name: string;
  when?: (ctx: CompileContext) => boolean;
  run: (ctx: CompileContext) => void;
};

export type RuntimeModuleLike = {
  run(funcName: string, ...args: unknown[]): unknown;
  runAsync(funcName: string, ...args: unknown[]): Promise<unknown>;
  isAsync(funcName: string): boolean;
  getKernelSource(funcName: string): string | null;
  getKernelMetadata(funcName: string): Record<string, unknown> | null;
  listKernels(): string[];
  executionPlan?: unknown;
};

export class CompilerConfig {
  target: CompileTarget;
  verify: VerifyLevelValue;
  errorMode: string;
  fusion: FusionConfig;
  scheduling: SchedulingConfig;
  matmulBackend: string;
  quantization: QuantizationConfig;
  optimization: OptimizationConfig;
  memory: MemoryConfig;
  partition: PartitionConfig;
  passContext: unknown;
  loweringRules: ReadonlyMap<string, unknown> | Record<string, unknown> | null;
  codegenEntries: ReadonlyMap<string, unknown> | Record<string, unknown> | null;
  trace: TraceConfig;

  constructor(opts: CompilerConfigOpts = {}) {
    this.target = opts.target as CompileTarget;
    this.verify = normalizeVerifyLevel(opts.verify);
    this.errorMode = opts.errorMode || 'strict';

    const schedulingDefaults = targetAttr<SchedulingDefaults>(this.target, TargetAttr.SCHEDULING) || {};

    this.fusion = { enabled: true, strategy: 'priority', ...opts.fusion };
    this.scheduling = { enabled: false, autotune: false, gpuTiling: false, ...schedulingDefaults, ...opts.scheduling };
    this.matmulBackend = opts.matmulBackend || 'native';
    this.quantization = { enabled: false, ...opts.quantization };
    this.optimization = {
      layout: false,
      rematerialization: false,
      rematConfig: {},
      fastMath: false,
      maxSimplifyIterations: 8,
      loopPartition: false,
      detectAccumulators: false,
      tensorize: false,
      splitSerializedKernels: true,
      ...opts.optimization,
    };
    this.memory = {
      alignment: 64,
      inplaceReuse: true,
      allocStrategy: 'best-fit',
      poolAllocation: false,
      scheduleForPeak: true,
      planReuse: true,
      planDonation: true,
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

  get usePartition(): boolean {
    return this.partition.enabled && this.partition.targets.length >= 2;
  }

  get verifyEnabled(): boolean {
    return this.verify !== VerifyLevel.OFF;
  }

  get verifyEachPass(): boolean {
    return this.verify === VerifyLevel.EACH_PASS;
  }
}

export class CompilationResult {
  module: RuntimeModuleLike;
  trace: TraceLog;
  errors: CompilationError[];

  constructor(runtimeModule: RuntimeModuleLike, trace: TraceLog, errors?: CompilationError[] | null) {
    this.module = runtimeModule;
    this.trace = trace;
    this.errors = errors || [];
  }

  get succeeded(): boolean { return this.errors.length === 0; }

  get failedFunctions(): Set<string> {
    const names = new Set<string>();
    for (const e of this.errors) if (e.funcName) names.add(e.funcName);
    return names;
  }

  run(funcName: string, ...args: unknown[]): unknown {
    return this.module.run(funcName, ...args);
  }

  async runAsync(funcName: string, ...args: unknown[]): Promise<unknown> {
    return this.module.runAsync(funcName, ...args);
  }

  isAsync(funcName: string): boolean {
    return this.module.isAsync(funcName);
  }

  getSource(funcName: string): string | null {
    return this.module.getKernelSource(funcName);
  }

  listKernels(): string[] {
    return this.module.listKernels();
  }
}

export class Compiler {
  config: CompilerConfig;
  context: CompilerContext;

  constructor(config: CompilerConfig | CompilerConfigOpts) {
    this.config = config instanceof CompilerConfig ? config : new CompilerConfig(config);
    if (!this.config.target) throw new Error('Compiler requires a target');
    this.context = new CompilerContext({
      loweringRules: this.config.loweringRules as Record<string, unknown> | null,
      codegenEntries: this.config.codegenEntries as Record<string, unknown> | null,
    });
  }

  compile(graphModule: GraphModule): CompilationResult {
    const trace = new TraceLog(this.config.trace as TraceLogConfig);
    const resilient = this.config.errorMode === 'resilient';
    const errors: CompilationError[] = [];
    const failed = new Set<string>();
    const t0 = performance.now();
    trace.phaseStart('compile');

    const ctx: CompileContext = {
      compiler: this,
      trace,
      errors,
      failed,
      resilient,
      original: graphModule,
      working: resilient ? cloneGraphModule(graphModule) : graphModule,
      split: null,
      tirModule: null,
      lirFuncs: null,
      runtimeModule: null,
      restartFrom: null,
    };

    const phases = this._compilePhases();
    let relaunches = 0;
    for (let i = 0; i < phases.length; i++) {
      const phase = phases[i];
      if (phase.when && !phase.when(ctx)) continue;
      phase.run(ctx);
      if (ctx.restartFrom === null) continue;

      const target = phases.findIndex(p => p.name === ctx.restartFrom);
      ctx.restartFrom = null;
      if (target < 0 || relaunches >= MAX_RELAUNCH_ATTEMPTS) continue;
      relaunches++;
      i = target - 1;
    }

    trace.phaseEnd('compile', performance.now() - t0);

    if (!resilient && errors.length > 0) {
      throw new Error(errors[0].toString());
    }

    return new CompilationResult(ctx.runtimeModule as RuntimeModuleLike, trace, errors);
  }

  _compilePhases(): CompilePhase[] {
    return [
      {
        name: 'verify:pre',
        when: (ctx: CompileContext) => ctx.compiler.config.verifyEnabled,
        run: (ctx: CompileContext) => ctx.compiler._verifyGraph(ctx.working, 'before graph passes', ctx.trace, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'graphPasses',
        run: (ctx: CompileContext) => ctx.compiler._runGraphPasses(ctx.working, ctx.original, ctx.trace, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'partition',
        when: (ctx: CompileContext) => ctx.compiler.config.usePartition,
        run: (ctx: CompileContext) => ctx.compiler._runPartitioning(ctx.working, ctx.trace),
      },
      {
        name: 'split',
        run: (ctx: CompileContext) => {
          const cfg = ctx.compiler.config;
          ctx.split = splitGraph(ctx.working, cfg, cfg.target) as GraphSplitResult;
        },
      },
      {
        name: 'verify:post',
        when: (ctx: CompileContext) => ctx.compiler.config.verifyEnabled,
        run: (ctx: CompileContext) => ctx.compiler._verifyGraph(ctx.working, 'after graph passes', ctx.trace, ctx.errors, ctx.failed, ctx.resilient),
      },
      {
        name: 'lowering',
        run: (ctx: CompileContext) => {
          const cfg = ctx.compiler.config;
          ctx.tirModule = ctx.compiler._lowerAll(ctx.working, ctx.trace, ctx.errors, ctx.failed, ctx.resilient);
          for (const provider of activeExternalCodegenProviders(cfg, cfg.target)) {
            if (provider.annotate) provider.annotate(ctx.tirModule, ctx.split);
          }
        },
      },
      {
        name: 'tirPasses',
        run: (ctx: CompileContext) => ctx.compiler._runTirPasses(ctx),
      },
      {
        name: 'verify:tensor',
        when: (ctx: CompileContext) => ctx.compiler.config.verifyEnabled,
        run: (ctx: CompileContext) => ctx.compiler._verifyBoundary(ctx.tirModule as TirModule, IRLevel.TIR, ctx),
      },
      {
        name: 'lirLowering',
        run: (ctx: CompileContext) => { ctx.lirFuncs = ctx.compiler._lowerToLIR(ctx.tirModule as TirModule, ctx.trace, ctx.errors, ctx.failed, ctx.resilient); },
      },
      {
        name: 'lirPasses',
        run: (ctx: CompileContext) => ctx.compiler._runLirPasses(ctx),
      },
      {
        name: 'verify:lir',
        when: (ctx: CompileContext) => ctx.compiler.config.verifyEnabled,
        run: (ctx: CompileContext) => ctx.compiler._verifyBoundary(ctx.lirFuncs as Iterable<{ name: string }>, IRLevel.LIR, ctx),
      },
      {
        name: 'codegen',
        run: (ctx: CompileContext) => {
          ctx.runtimeModule = ctx.compiler._codegen(ctx.lirFuncs as LIRFuncLike[], ctx.trace, ctx.errors, ctx.failed, ctx.resilient);
          if (ctx.split) (ctx.runtimeModule as RuntimeModuleLike).executionPlan = ctx.split.plan;
        },
      },
      {
        name: 'relaunchOnSerialization',
        run: (ctx: CompileContext) => ctx.compiler._relaunchOnSerialization(ctx),
      },
      {
        name: 'planBufferAssignment',
        when: (ctx: CompileContext) => ctx.compiler.config.memory.planReuse !== false && !!(ctx.split && ctx.split.plan),
        run: (ctx: CompileContext) => ctx.compiler._assignPlanBuffers(ctx),
      },
    ];
  }

  compileFunction(graphFunc: GraphFunction): CompilationResult {
    const mod = new GraphModule('single');
    mod.addFunction(graphFunc);
    return this.compile(mod);
  }

  calibrate(graphModule: GraphModule, mode = 'minmax'): unknown {
    const collector = new CalibrationCollector(mode);
    for (const func of graphModule) {
      collector.attach(func);
    }
    return collector;
  }

  _runGraphPasses(graphModule: GraphModule, original: GraphModule, trace: TraceLog, errors: CompilationError[], failed: Set<string>, resilient: boolean): void {
    const pm = new PassManager();

    const compileFn: CompileFn = (mod, tgt) => new Compiler({ target: tgt as CompileTarget, verify: this.config.verify }).compile(mod);
    for (const p of buildGraphPipeline(this.config, this.config.target, { context: this.context, compileFn })) {
      pm.addPass(p);
    }

    pm.setTrace(trace);
    pm.setCheckEachPass(this.config.verifyEachPass);

    trace.phaseStart('graphPasses');
    const t0 = performance.now();
    const result = pm.run(graphModule, { errorMode: resilient ? 'resilient' : 'strict', passContext: this.config.passContext as PassContext | null });
    if (result.errors) {
      for (const e of result.errors as CompilationError[]) {
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
  }

  _runPartitioning(graphModule: GraphModule, trace: TraceLog): void {
    const pm = new PassManager();
    pm.addPass(new GraphPartitionPass(this.config.partition));
    pm.addPass(new PartitionMaterializationPass());

    pm.setTrace(trace);

    trace.phaseStart('partition');
    const t0 = performance.now();
    pm.run(graphModule);
    trace.phaseEnd('partition', performance.now() - t0);
  }

  _eachFunc<T extends { name: string }>(funcs: Iterable<T>, phase: string, trace: TraceLog, errors: CompilationError[], failed: Set<string>, resilient: boolean, fn: (f: T) => void): void {
    for (const f of funcs) {
      if (failed.has(f.name)) continue;
      try {
        fn(f);
      } catch (e) {
        const msg = (e as Error).message;
        errors.push(new CompilationError(phase, f.name, msg));
        failed.add(f.name);
        trace.errorEvent(phase, f.name, msg);
        if (!resilient) break;
      }
    }
  }

  _lowerAll(graphModule: GraphModule, trace: TraceLog, errors: CompilationError[], failed: Set<string>, resilient: boolean): TirModule {
    trace.phaseStart('lowering');
    const t0 = performance.now();
    const tirModule = new TirModule(graphModule.name);
    this._eachFunc(graphModule, 'lowering', trace, errors, failed, resilient, (func) => {
      const ft0 = performance.now();
      const primFunc = lowerGraphToPrimFunc(func, this.config.target as unknown as null, this.context as unknown as null);
      if (this.config.target.isGPU && this.config.target.isGPU() && !(this.config.target.isWebGPU && this.config.target.isWebGPU())) {
        const convInfo = detectPureConv(func);
        if (convInfo) primFunc.setAttr(FuncAttr.CONV_INFO, convInfo);
      }
      trace.functionEvent('lowering', func.name, { durationMs: performance.now() - ft0 });
      tirModule.addFunction(primFunc);
      if (trace.shouldSnapshot('afterLowering')) {
        trace.irDump('afterLowering:' + func.name, printTensorIR(primFunc));
      }
    });
    trace.phaseEnd('lowering', performance.now() - t0);
    return tirModule;
  }

  _runTirPasses(ctx: CompileContext): void {
    const tirPM = new TirPassManager();
    for (const pass of buildTirPipeline(this.config)) tirPM.addPass(pass);
    tirPM.setTrace(ctx.trace);
    tirPM.setCheckEachPass(this.config.verifyEachPass);
    tirPM.run(ctx.tirModule as TirModule, {
      trace: ctx.trace,
      errors: ctx.errors,
      failed: ctx.failed,
      resilient: ctx.resilient,
    });
  }

  _serializedKernels(runtimeModule: RuntimeModuleLike): { name: string; diagnosis: GpuLaunchDiagnosis }[] {
    const serialized: { name: string; diagnosis: GpuLaunchDiagnosis }[] = [];
    for (const name of runtimeModule.listKernels()) {
      const metadata = runtimeModule.getKernelMetadata(name);
      const diagnosis = metadata ? metadata.launchDiagnosis as GpuLaunchDiagnosis | null : null;
      if (diagnosis) serialized.push({ name, diagnosis });
    }
    return serialized;
  }

  _hasSequentialRegion(module: GraphModule): boolean {
    for (const func of module) {
      for (const op of func.ops()) {
        if (containsSequentialRegion(op)) return true;
      }
    }
    return false;
  }

  _relaunchOnSerialization(ctx: CompileContext): void {
    if (this.config.optimization.splitSerializedKernels === false) return;
    const serialized = this._serializedKernels(ctx.runtimeModule as RuntimeModuleLike);
    if (serialized.length === 0) return;

    const module = ctx.working;
    const detail = {
      kernels: serialized.map(s => s.name),
      reasons: [...new Set(serialized.map(s => s.diagnosis.reason))],
      buffers: [...new Set(serialized.flatMap(s => s.diagnosis.buffers))],
    };

    if (ctx.failed.size > 0) {
      ctx.trace.warn('relaunch', module.name, 'keeping serialized kernel: earlier phases already failed', detail);
      return;
    }

    if (this._hasSequentialRegion(module)) {
      ctx.trace.warn('relaunch', module.name,
        'keeping serialized kernel: a sequential recurrence must stay in one kernel', detail);
      return;
    }

    const split = splitGraphForNative(module, RACE_SPLIT_MIN_BOUNDARIES) as GraphSplitResult;
    if (!split) {
      ctx.trace.warn('relaunch', module.name, 'keeping serialized kernel: the graph cannot be split any further', detail);
      return;
    }

    ctx.trace.explain('relaunch', module.name, 'split into separate kernels and recompiled',
      'the fused kernel could not run at its launch geometry', detail);
    ctx.split = split;
    ctx.tirModule = null;
    ctx.lirFuncs = null;
    ctx.runtimeModule = null;
    ctx.restartFrom = 'lowering';
  }

  _assignPlanBuffers(ctx: CompileContext): void {
    const plan = (ctx.split as { plan: AssignablePlan }).plan;
    const donations = this.config.memory.planDonation === false ? [] : computePlanDonations(ctx.working, plan);
    const buffers = assignPlanBuffers(plan, donations);
    if (!buffers) return;
    plan.buffers = buffers;
    ctx.trace.memoryStats(ctx.working.name, { plan: planMemoryReport(plan, buffers), slots: plan.numSlots });
  }

  _runLirPasses(ctx: CompileContext): void {
    const pm = new LirPassManager();
    for (const pass of buildLirPipeline(this.config)) pm.addPass(pass);
    pm.setTrace(ctx.trace);
    pm.setCheckEachPass(this.config.verifyEachPass);
    pm.run(ctx.lirFuncs as LIRFunc[], {
      trace: ctx.trace,
      config: this.config,
      errors: ctx.errors,
      failed: ctx.failed,
      resilient: ctx.resilient,
    });
  }

  _verifyGraph(graphModule: GraphModule, phase: string, trace: TraceLog, errors: CompilationError[], failed: Set<string>, resilient: boolean): void {
    if (resilient) {
      for (const func of graphModule) {
        if (failed.has(func.name)) continue;
        const err = checkIRInvariants(IRLevel.GRAPH_FUNC, func, func.name);
        if (!err) continue;
        errors.push(err);
        failed.add(func.name);
        trace.errorEvent('verification', func.name, err.message);
      }
      return;
    }
    const err = checkIRInvariants(IRLevel.GRAPH_MODULE, graphModule, graphModule.name || '<module>');
    if (err) throw new Error('Graph verification failed (' + phase + '): ' + err.message);
  }

  _verifyBoundary(funcs: Iterable<{ name: string }>, irLevel: IRLevelValue, ctx: CompileContext): void {
    for (const func of funcs) {
      if (ctx.failed.has(func.name)) continue;
      const err = checkIRInvariants(irLevel, func, func.name);
      if (!err) continue;
      if (!ctx.resilient) throw new Error(err.toString());
      ctx.errors.push(err);
      ctx.failed.add(func.name);
      ctx.trace.errorEvent('verification', func.name, err.message);
    }
  }

  _lowerToLIR(tirModule: TirModule, trace: TraceLog, errors: CompilationError[], failed: Set<string>, resilient: boolean): LIRFuncLike[] {
    trace.phaseStart('lirLowering');
    const t0 = performance.now();
    const lirFuncs: LIRFuncLike[] = [];
    this._eachFunc(tirModule, 'lirLowering', trace, errors, failed, resilient, (pf) => {
      const ft0 = performance.now();
      const lirFunc = lowerToLIR(pf, this.config.target as CompileTarget) as unknown as LIRFuncLike;
      trace.functionEvent('lirLowering', pf.name, { durationMs: performance.now() - ft0 });
      lirFuncs.push(lirFunc);
    });
    trace.phaseEnd('lirLowering', performance.now() - t0);
    return lirFuncs;
  }

  _codegen(primFuncs: readonly LIRFuncLike[], trace: TraceLog, errors: CompilationError[], failed: Set<string>, resilient: boolean): RuntimeModuleLike {
    trace.phaseStart('codegen');
    const t0 = performance.now();
    const runtimeMod = new RuntimeModule('compiled');

    const usePartition = this.config.usePartition;
    const backendOpts = { context: this.context };
    const backendCache = new Map<string, InstanceType<typeof BackendPipeline>>();
    const getBackend = (target: CompileTarget) => {
      if (!backendCache.has(target.name)) backendCache.set(target.name, new BackendPipeline(target, backendOpts));
      return backendCache.get(target.name) as InstanceType<typeof BackendPipeline>;
    };
    const defaultBackend = usePartition ? null : new BackendPipeline(this.config.target, backendOpts);

    this._eachFunc(primFuncs, 'codegen', trace, errors, failed, resilient, (pf) => {
      const ft0 = performance.now();
      let backend: InstanceType<typeof BackendPipeline>;
      if (usePartition) {
        const targetName = (pf.getAttr as (k: string) => unknown)(FuncAttr.PARTITION_TARGET);
        const target = targetName
          ? (this.config.partition.targets as CompileTarget[]).find(t => t.name === targetName)
          : this.config.target;
        backend = getBackend((target || this.config.target) as CompileTarget);
      } else {
        backend = defaultBackend as InstanceType<typeof BackendPipeline>;
      }
      const compiled = backend.compile(pf as unknown as PrimFunc);
      runtimeMod.addCompiledKernel(compiled as never);
      if (pf.shapeParamMap && pf.shapeParamMap.size > 0) {
        runtimeMod.setShapeParamMap(pf.name, pf.shapeParamMap as never, pf.bufferMap as never);
      }
      trace.codegenStats(pf.name, {
        durationMs: performance.now() - ft0,
        sourceSize: compiled.source.length,
        targetName: compiled.target.name,
      });
      const diagnosis = compiled.metadata.launchDiagnosis as GpuLaunchDiagnosis | null | undefined;
      if (diagnosis) {
        trace.warn('codegen', pf.name, `kernel serialized to a single thread: ${diagnosis.reason}`, diagnosis);
      }
    });

    trace.phaseEnd('codegen', performance.now() - t0);
    return runtimeMod;
  }
}

export function compileGraph(graphFunc: GraphFunction, target: CompileTarget, opts: CompilerConfigOpts = {}): CompilationResult {
  const compiler = new Compiler({ target, ...opts });
  return compiler.compileFunction(graphFunc);
}
