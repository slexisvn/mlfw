import { compile, compileWithBackward, manual_seed } from 'mlfw/index.js';
import { asCompilable, compilerOptions } from './settings.js';
import { evaluateModelSource } from './evaluate.js';
import { executeCompiled } from './execute.js';
import { ddmin } from './ddmin.js';
import type { BisectMode, BisectProbe, BisectRequest, BisectResponse, CompileOptions, IRLevelName, RunResult } from '../protocol.js';

const SEED = 0;

type InferenceHandle = { _ready: Promise<void> | null };

type PassSeen = { name: string; level: IRLevelName };

class PassCollector {
  readonly seen: PassSeen[] = [];
  private readonly names = new Set<string>();

  runBeforePass = (pass: { name: string }, _target: unknown, level: IRLevelName): void => {
    if (this.names.has(pass.name)) return;
    this.names.add(pass.name);
    this.seen.push({ name: pass.name, level });
  };
}

function worstDiff(run: RunResult): number | null {
  if (run.maxAbsDiff === null) return run.maxAbsGradDiff;
  if (run.maxAbsGradDiff === null) return run.maxAbsDiff;
  return Math.max(run.maxAbsDiff, run.maxAbsGradDiff);
}

function isGood(probe: Omit<BisectProbe, 'good' | 'index'>, tolerance: number): boolean {
  if (!probe.ok || probe.error !== null) return false;
  if (!probe.ran) return true;
  return probe.diff !== null && probe.diff <= tolerance;
}

async function runProbe(
  source: string,
  options: CompileOptions,
  disabled: readonly string[],
  tolerance: number,
  collector: PassCollector | null,
): Promise<Omit<BisectProbe, 'index'>> {
  const startedAt = performance.now();
  let ok = false;
  let error: string | null = null;
  let run: RunResult | null = null;
  let stopRecordingLines = (): void => {};

  try {
    manual_seed(SEED);
    const { model, inputs } = evaluateModelSource(source, stop => { stopRecordingLines = stop; });
    stopRecordingLines();
    const compilable = asCompilable(model);
    const settings = {
      ...compilerOptions(options, disabled),
      instruments: collector ? [collector] : [],
    };

    const handle = options.backward === 'off'
      ? compile(compilable, inputs as never[], settings as never)
      : compileWithBackward(compilable, inputs as never[], { ...settings, mode: options.backward } as never);

    const ready = (handle as unknown as InferenceHandle)._ready;
    if (ready) await ready;
    ok = true;

    manual_seed(SEED);
    run = await executeCompiled(
      handle as (...args: unknown[]) => unknown,
      compilable,
      inputs,
      options.target,
      options.backward,
      { quick: true },
    );
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  } finally {
    stopRecordingLines();
  }

  const probe = {
    disabled: [...disabled],
    ok,
    ran: run !== null && run.ran,
    error: error ?? (run === null ? null : run.error),
    diff: run === null ? null : worstDiff(run),
    ms: performance.now() - startedAt,
  };

  return { ...probe, good: isGood(probe, tolerance) };
}

function conclusionFor(
  mode: BisectMode,
  culprits: readonly string[],
  candidates: readonly string[],
  tolerance: number,
): string {
  const broke = mode === 'compile' ? 'the compile threw' : `the answer drifted past ${tolerance}`;

  if (culprits.length === 0) {
    return `Nothing to blame: ${broke} even with every one of the ${candidates.length} passes turned off, so the fault is in tracing, lowering or codegen — not in a pass.`;
  }

  if (culprits.length === 1) {
    return `${culprits[0]} is the culprit: turning that one pass off is enough to fix it, and turning it back on breaks it again.`;
  }

  return `${culprits.join(' + ')} break it together: no smaller subset of the ${candidates.length} passes that ran is enough, each of these has to be off.`;
}

export async function bisect(request: BisectRequest, report: (probe: BisectProbe, note: string) => void): Promise<BisectResponse> {
  const startedAt = performance.now();
  const { id, source, options, tolerance } = request;
  const probes: BisectProbe[] = [];

  const record = (probe: Omit<BisectProbe, 'index'>, note: string): BisectProbe => {
    const full = { ...probe, index: probes.length };
    probes.push(full);
    report(full, note);
    return full;
  };

  const answer = (fields: Partial<BisectResponse>): BisectResponse => ({
    kind: 'bisect',
    id,
    mode: null,
    tolerance,
    baseline: probes[0] ?? null,
    allOff: null,
    candidates: [],
    culprits: [],
    probes,
    conclusion: '',
    error: null,
    totalMs: performance.now() - startedAt,
    ...fields,
  });

  const collector = new PassCollector();
  const baseline = record(
    await runProbe(source, options, [], tolerance, collector),
    'compiling with everything on, to see what is wrong and which passes run',
  );

  if (baseline.good) {
    return answer({
      conclusion: baseline.ran
        ? `Nothing to bisect: this compile already agrees with eager to within ${tolerance}.`
        : 'Nothing to bisect: this compile succeeded, and this target cannot run here to be checked against eager.',
    });
  }

  const mode: BisectMode = baseline.ok ? 'numeric' : 'compile';
  const candidates = collector.seen.map(pass => pass.name);

  if (candidates.length === 0) {
    return answer({
      mode,
      conclusion: 'Nothing to bisect: no pass ran at all, so this broke before the pipeline started.',
      error: baseline.error,
    });
  }

  const tried = new Map<string, boolean>();
  const test = async (disabled: readonly string[]): Promise<boolean> => {
    const key = [...disabled].sort().join('|');
    const cached = tried.get(key);
    if (cached !== undefined) return cached;

    const probe = record(
      await runProbe(source, options, disabled, tolerance, null),
      `${disabled.length} of ${candidates.length} passes off`,
    );
    tried.set(key, probe.good);
    return probe.good;
  };

  const allOffGood = await test(candidates);
  const allOff = probes[probes.length - 1];

  if (!allOffGood) {
    return answer({
      mode,
      allOff,
      candidates,
      conclusion: conclusionFor(mode, [], candidates, tolerance),
      error: baseline.error,
    });
  }

  const culprits = await ddmin(candidates, test);

  return answer({
    mode,
    allOff,
    candidates,
    culprits,
    conclusion: conclusionFor(mode, culprits, candidates, tolerance),
    error: baseline.error,
  });
}
