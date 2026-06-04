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
import { BackendPipeline } from '../backend/pipeline.js';
import { RuntimeModule } from '../runtime/runtime.js';
import { Autotuner } from '../autotune/autotuner.js';
import { TensorVerifier } from '../ir/tensor/verifier.js';
import { verifyModule } from '../ir/verifier/verifier.js';
import { CalibrationCollector } from '../analysis/calibration.js';
import { DecompositionPass } from '../passes/decompose/decomposition_pass.js';
import { RematerializationPass } from '../passes/memory/rematerialization.js';

export class CompilerConfig {
  constructor(opts = {}) {
    this.target = opts.target;
    this.verify = opts.verify !== false;
    this.enableDiagnostics = opts.enableDiagnostics || false;

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
  }
}

function spread(obj) { return obj && typeof obj === 'object' ? obj : {}; }
function omit(obj, ...keys) {
  const s = new Set(keys);
  const r = {};
  for (const k of Object.keys(obj)) { if (!s.has(k)) r[k] = obj[k]; }
  return r;
}

export class CompilationResult {
  constructor(runtimeModule, diagnostics) {
    this.module = runtimeModule;
    this.diagnostics = diagnostics;
  }

  run(funcName, ...args) {
    return this.module.run(funcName, ...args);
  }

  getSource(funcName) {
    return this.module.getKernelSource(funcName);
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
    const diag = this.config.enableDiagnostics ? new CompilerDiagnostics(this.config.target.name) : null;

    if (this.config.verify) {
      this._verifyGraph(graphModule, 'before graph passes');
    }

    this._runGraphPasses(graphModule, diag);

    if (this.config.verify) {
      this._verifyGraph(graphModule, 'after graph passes');
    }

    const primFuncs = this._lowerAll(graphModule, diag);

    this._scheduleAll(primFuncs, diag);

    const memoryPlans = this._planMemory(primFuncs, diag);

    if (this.config.verify) {
      this._verifyAll(primFuncs, diag);
    }

    const runtimeModule = this._codegen(primFuncs, diag);

    return new CompilationResult(runtimeModule, diag);
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

  _runGraphPasses(graphModule, diag) {
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

    if (diag) diag.record('graphPasses', 'start');
    const result = pm.run(graphModule);
    if (diag) diag.record('graphPasses', 'done', { changed: result.changed });
  }

  _lowerAll(graphModule, diag) {
    const primFuncs = [];
    for (const func of graphModule) {
      if (diag) diag.record('lowering', func.name);
      const primFunc = lowerGraphToPrimFunc(func);
      primFuncs.push(primFunc);
    }
    return primFuncs;
  }

  _scheduleAll(primFuncs, diag) {
    const sCfg = this.config.scheduling;
    if (sCfg.autotune) {
      const autotuner = new Autotuner(this.config.target, sCfg);
      for (const pf of primFuncs) {
        if (diag) diag.record('autotune', pf.name);
        autotuner.tuneAndApply(pf);
      }
    } else if (sCfg.enabled) {
      const policy = new SchedulePolicy(this.config.target);
      for (const pf of primFuncs) {
        if (diag) diag.record('schedule', pf.name);
        const sch = new Schedule(pf);
        policy.applyToAllBlocks(sch);
      }
    }
  }

  _planMemory(primFuncs, diag) {
    const alignment = this.config.memory.alignment
      || this.config.target?.cacheLineSizeBytes
      || 64;
    const planner = new MemoryPlanner({
      alignment,
      enableInplace: this.config.memory.inplaceReuse
    });
    const plans = [];
    for (const pf of primFuncs) {
      if (diag) diag.record('memoryPlan', pf.name);
      const { plan } = planner.planAndRewrite(pf);
      plans.push(plan);
      if (diag) diag.record('memoryPlan', pf.name, { peak: plan.peakMemory() });
    }
    return plans;
  }

  _verifyGraph(graphModule, phase) {
    const errors = verifyModule(graphModule);
    if (errors.length > 0) {
      throw new Error(`Graph verification failed (${phase}): ${errors.map(e => e.toString()).join('; ')}`);
    }
  }

  _verifyAll(primFuncs, diag) {
    const verifier = new TensorVerifier();
    for (const pf of primFuncs) {
      const errors = verifier.verify(pf);
      if (errors.length > 0) {
        throw new Error(`TensorIR verification failed for ${pf.name}: ${errors.join('; ')}`);
      }
    }
  }

  _codegen(primFuncs, diag) {
    const backend = new BackendPipeline(this.config.target);
    const runtimeMod = new RuntimeModule('compiled');

    for (const pf of primFuncs) {
      if (diag) diag.record('codegen', pf.name);
      const compiled = backend.compile(pf);
      runtimeMod.addCompiledKernel(compiled);
    }

    return runtimeMod;
  }
}

class CompilerDiagnostics {
  constructor(targetName) {
    this.targetName = targetName || 'unknown';
    this.entries = [];
  }

  record(phase, detail, extra = null) {
    this.entries.push({ phase, detail, extra, timestamp: Date.now() });
  }

  getPhase(phase) {
    return this.entries.filter(e => e.phase === phase);
  }

  summary() {
    const phases = new Map();
    for (const e of this.entries) {
      if (!phases.has(e.phase)) phases.set(e.phase, []);
      phases.get(e.phase).push(e);
    }
    return phases;
  }
}

export function compileGraph(graphFunc, target, opts = {}) {
  const compiler = new Compiler({ target, ...opts });
  return compiler.compileFunction(graphFunc);
}
