import { useSyncExternalStore } from 'react';
import { DEFAULT_OPTIONS } from './protocol.js';
import { EXAMPLES } from './examples/index.js';
import { readSession, writeSession } from './session.js';
import { provenanceOf } from './catalog/provenance.js';
import type { Ledger } from './catalog/ledger.js';
import type { BisectProbe, BisectResponse, CompileOptions, CompileResponse, CompileStep, SemanticsResponse, WorkerMessage, WorkerProgress, WorkerRequest, WorkerRequestDraft, WorkerResponse } from './protocol.js';
import type { StageTab } from './catalog/glossary.js';

const REDUCED_MOTION = typeof matchMedia === 'function'
  && matchMedia('(prefers-reduced-motion: reduce)').matches;

const RUNS: ReadonlySet<string> = new Set(['pass', 'primitive']);

export const SPEEDS = [0, 0.5, 1, 2] as const;
export type Status = 'starting' | 'idle' | 'compiling' | 'ready' | 'failed';
export type Pane = 'source' | 'timeline' | 'stage';

export type Failure = { error: string; errorPhase: string | null };
export type Baseline = { response: CompileResponse; options: CompileOptions; source: string };

export type State = {
  source: string;
  exampleId: string;
  options: CompileOptions;
  globals: string[];
  status: Status;
  result: CompileResponse | null;
  baseline: Baseline | null;
  failure: Failure | null;
  ranSource: string | null;
  ranOptions: CompileOptions | null;
  passPhases: Record<string, string>;
  selected: number;
  tab: StageTab;
  speed: number;
  playing: boolean;
  onlyChanged: boolean;
  collapsed: ReadonlySet<number>;
  pane: Pane;
  focusLine: number | null;
  bisecting: boolean;
  bisect: BisectResponse | null;
  bisectProbes: BisectProbe[];
  bisectNote: string;
  semantics: Record<number, SemanticsResponse>;
  semanticsPending: number | null;
  find: string;
  ledger: Ledger | null;
};

const worker = new Worker(new URL('./worker/compile.worker.ts', import.meta.url), { type: 'module' });

type Pending = {
  resolve: (response: WorkerResponse) => void;
  onProgress: ((progress: WorkerProgress) => void) | null;
};

let nextRequestId = 1;
const pending = new Map<number, Pending>();

worker.onmessage = (message: MessageEvent<WorkerMessage>) => {
  const data = message.data;
  const entry = pending.get(data.id);
  if (!entry) return;

  if (data.kind === 'bisect-progress') {
    if (entry.onProgress) entry.onProgress(data);
    return;
  }

  pending.delete(data.id);
  entry.resolve(data);
};

function ask(request: WorkerRequestDraft, onProgress: ((progress: WorkerProgress) => void) | null = null): Promise<WorkerResponse> {
  const id = nextRequestId++;
  return new Promise(resolve => {
    pending.set(id, { resolve, onProgress });
    worker.postMessage({ ...request, id } as WorkerRequest);
  });
}

const restored = readSession();

let state: State = {
  source: restored ? restored.source : EXAMPLES[0].source,
  exampleId: restored ? restored.exampleId : EXAMPLES[0].id,
  options: restored ? restored.options : DEFAULT_OPTIONS,
  globals: [],
  status: 'starting',
  result: null,
  baseline: null,
  failure: null,
  ranSource: null,
  ranOptions: null,
  passPhases: {},
  selected: 0,
  tab: 'ir',
  speed: REDUCED_MOTION ? 0 : 1,
  playing: false,
  onlyChanged: true,
  collapsed: new Set(),
  pane: 'source',
  focusLine: null,
  bisecting: false,
  bisect: null,
  bisectProbes: [],
  bisectNote: '',
  semantics: {},
  semanticsPending: null,
  find: '',
  ledger: null,
};

const listeners = new Set<() => void>();

function set(patch: Partial<State>): void {
  state = { ...state, ...patch };
  for (const listener of listeners) listener();
}

function persist(): void {
  writeSession({ source: state.source, exampleId: state.exampleId, options: state.options });
}

export function useStore<T>(select: (s: State) => T): T {
  return useSyncExternalStore(
    listener => { listeners.add(listener); return () => listeners.delete(listener); },
    () => select(state),
  );
}

export function getState(): State {
  return state;
}

const familyCache = new WeakMap<CompileResponse, Family>();

export type Family = { ownerOf: Map<number, number>; childrenOf: Map<number, number[]> };

export function family(response: CompileResponse): Family {
  const cached = familyCache.get(response);
  if (cached) return cached;

  const ownerOf = new Map<number, number>();
  const childrenOf = new Map<number, number[]>();
  let owner: number | null = null;

  for (const step of response.steps) {
    if (step.kind !== 'primitive') {
      owner = step.kind === 'pass' ? step.index : null;
      continue;
    }
    if (owner === null) continue;
    ownerOf.set(step.index, owner);
    const kids = childrenOf.get(owner);
    if (kids) kids.push(step.index);
    else childrenOf.set(owner, [step.index]);
  }

  const computed = { ownerOf, childrenOf };
  familyCache.set(response, computed);
  return computed;
}

export function childCount(s: State, index: number): number {
  if (!s.result) return 0;
  return (family(s.result).childrenOf.get(index) ?? []).length;
}

export function isCollapsed(s: State, index: number): boolean {
  return s.collapsed.has(index);
}

function derived<T>(deps: (s: State) => readonly unknown[], compute: (s: State) => T): (s: State) => T {
  let last: readonly unknown[] | null = null;
  let value: T;
  return (s: State) => {
    const next = deps(s);
    if (last !== null && last.length === next.length && last.every((dep, i) => dep === next[i])) return value;
    last = next;
    value = compute(s);
    return value;
  };
}

function collapsedByDefault(response: CompileResponse): Set<number> {
  return new Set(family(response).childrenOf.keys());
}

export const visibleSteps = derived(
  s => [s.result, s.onlyChanged, s.collapsed],
  (s): CompileStep[] => {
    const computed = !s.result ? [] : s.result.steps.filter(step => shown(s, step));
    return computed.length > 0 || !s.result ? computed : s.result.steps;
  },
);

function quiet(s: State, step: CompileStep | undefined): boolean {
  return !!step && s.onlyChanged && step.outcome === 'unchanged' && RUNS.has(step.kind);
}

function shown(s: State, step: CompileStep): boolean {
  if (quiet(s, step)) return false;
  if (step.kind !== 'primitive') return true;

  const result = s.result as CompileResponse;
  const owner = family(result).ownerOf.get(step.index);
  if (owner === undefined) return true;
  if (s.collapsed.has(owner)) return false;

  return !quiet(s, result.steps[owner]);
}

export type RunTally = { total: number; changed: number; quiet: number };

const NO_RUNS: RunTally = { total: 0, changed: 0, quiet: 0 };
const tallyCache = new WeakMap<CompileResponse, RunTally>();

export function passRunCount(s: State): RunTally {
  if (!s.result) return NO_RUNS;
  const cached = tallyCache.get(s.result);
  if (cached) return cached;

  let total = 0;
  let changed = 0;
  let quiet = 0;
  for (const step of s.result.steps) {
    if (step.kind === 'pass') {
      total++;
      if (step.outcome === 'changed') changed++;
    }
    if (RUNS.has(step.kind) && step.outcome === 'unchanged') quiet++;
  }

  const tally = { total, changed, quiet };
  tallyCache.set(s.result, tally);
  return tally;
}

export function isStale(s: State): boolean {
  if (!s.result || s.ranOptions === null) return false;
  if (s.ranSource !== s.source) return true;
  const keys = Object.keys(s.options) as (keyof CompileOptions)[];
  return keys.some(key => JSON.stringify(s.options[key]) !== JSON.stringify((s.ranOptions as CompileOptions)[key]));
}

const NO_LINES: readonly number[] = [];

export function sourceLines(s: State): readonly number[] {
  return s.result ? s.result.sourceLines : NO_LINES;
}

export const provenance = derived(
  s => [s.result, s.find],
  s => (s.result ? provenanceOf(s.result, s.find) : null),
);

const NO_STEPS: readonly CompileStep[] = [];

export const interpretableSteps = derived(
  s => [s.result],
  (s): readonly CompileStep[] => (s.result ? s.result.steps.filter(step => step.interpretable) : NO_STEPS),
);

export type DisabledStatus = 'off' | 'ignored' | 'pending';

export type DisabledPass = { name: string; phase: string; status: DisabledStatus };

function disabledStatus(s: State, name: string, skipped: ReadonlySet<string>): DisabledStatus {
  if (s.result === null || isStale(s)) return 'pending';
  return skipped.has(name) ? 'off' : 'ignored';
}

export const disabledPasses = derived(
  s => [s.options, s.passPhases, s.result, s.ranOptions, s.ranSource, s.source],
  (s): DisabledPass[] => {
    const skipped = new Set((s.result ? s.result.skipped : []).map(entry => entry.pass));
    return s.options.disabledPasses.map(name => ({
      name,
      phase: s.passPhases[name] ?? 'turned off',
      status: disabledStatus(s, name, skipped),
    }));
  },
);

function nearestVisible(s: State, index: number): number {
  const steps = visibleSteps(s);
  if (steps.length === 0) return 0;
  let best = steps[0];
  for (const step of steps) {
    if (Math.abs(step.index - index) < Math.abs(best.index - index)) best = step;
  }
  return best.index;
}

function firstSuspect(response: CompileResponse): number | null {
  const broke = response.steps.find(step => step.verify !== null && step.verify.introduced.length > 0);
  if (broke) return broke.index;
  const failed = response.steps.find(step => step.outcome === 'failed');
  return failed ? failed.index : null;
}

function phasesOf(response: CompileResponse, previous: Record<string, string>): Record<string, string> {
  const merged = { ...previous };
  for (const step of response.steps) {
    if (step.kind === 'pass') merged[step.pass] = step.phase;
  }
  return merged;
}

export const actions = {
  async init(): Promise<void> {
    const response = await ask({ kind: 'init' }) as { globals: string[] };
    set({ globals: response.globals, status: 'idle' });
  },

  setSource(source: string): void {
    set({ source, exampleId: '', focusLine: null });
    persist();
  },

  loadExample(id: string): void {
    const example = EXAMPLES.find(e => e.id === id);
    if (!example) return;
    set({ source: example.source, exampleId: id, focusLine: null });
    persist();
  },

  setOptions(patch: Partial<CompileOptions>): void {
    set({ options: { ...state.options, ...patch } });
    persist();
  },

  select(index: number): void {
    const steps = visibleSteps(state);
    if (steps.length === 0) return;
    const clamped = Math.max(0, Math.min((state.result as CompileResponse).steps.length - 1, index));
    const last = steps[steps.length - 1].index;
    set({ selected: clamped, playing: state.playing && clamped < last, pane: 'stage' });
  },

  step(delta: number): void {
    const steps = visibleSteps(state);
    if (steps.length === 0) return;
    const at = steps.findIndex(s => s.index === state.selected);
    const next = steps[Math.max(0, Math.min(steps.length - 1, (at < 0 ? 0 : at) + delta))];
    actions.select(next.index);
  },

  toggleCollapse(index: number): void {
    if (!state.result) return;
    const collapsed = new Set(state.collapsed);
    if (collapsed.has(index)) collapsed.delete(index);
    else collapsed.add(index);

    const next = { ...state, collapsed };
    const owner = family(state.result).ownerOf.get(state.selected);
    const hidden = owner !== undefined && collapsed.has(owner);
    set({
      collapsed,
      selected: hidden ? owner : nearestVisible(next, state.selected),
      playing: false,
    });
  },

  toggleOnlyChanged(): void {
    const next = { ...state, onlyChanged: !state.onlyChanged };
    set({ onlyChanged: next.onlyChanged, selected: nearestVisible(next, state.selected), playing: false });
  },

  pinBaseline(): void {
    if (!state.result) return;
    set({ baseline: { response: state.result, options: state.ranOptions ?? state.options, source: state.ranSource ?? state.source } });
  },

  clearBaseline(): void {
    set({ baseline: null });
  },

  setTab(tab: StageTab): void {
    set({ tab, pane: 'stage' });
  },

  reveal(index: number, tab: StageTab): void {
    const step = state.result ? state.result.steps[index] : undefined;
    if (!step) return;

    const owner = state.result ? family(state.result).ownerOf.get(index) : undefined;
    const collapsed = owner !== undefined && state.collapsed.has(owner)
      ? new Set([...state.collapsed].filter(entry => entry !== owner))
      : state.collapsed;

    set({
      selected: index,
      onlyChanged: quiet(state, step) ? false : state.onlyChanged,
      collapsed,
      tab,
      pane: 'stage',
      playing: false,
    });
  },

  setLedger(ledger: Ledger | null): void {
    set({ ledger });
  },

  setFind(find: string): void {
    if (state.find !== find) set({ find });
  },

  setPane(pane: Pane): void {
    set({ pane });
  },

  setSpeed(speed: number): void {
    set({ speed });
  },

  togglePlay(): void {
    if (!state.result || state.result.steps.length === 0) return;
    const atEnd = state.selected >= state.result.steps.length - 1;
    set({ playing: !state.playing, selected: !state.playing && atEnd ? 0 : state.selected });
  },

  stopPlay(): void {
    if (state.playing) set({ playing: false });
  },

  async runBisect(tolerance: number): Promise<void> {
    if (state.bisecting) return;
    set({ bisecting: true, bisect: null, bisectProbes: [], bisectNote: 'compiling once with everything on…', tab: 'bisect', pane: 'stage' });

    const response = await ask(
      { kind: 'bisect', source: state.source, options: state.options, tolerance },
      progress => set({ bisectProbes: [...state.bisectProbes, progress.probe], bisectNote: progress.note }),
    ) as BisectResponse;

    set({ bisecting: false, bisect: response, bisectProbes: response.probes, bisectNote: '' });
  },

  async proveStep(index: number): Promise<void> {
    if (state.semanticsPending !== null || state.semantics[index]) return;
    set({ semanticsPending: index });
    const response = await ask({ kind: 'semantics', step: index }) as SemanticsResponse;
    set({ semantics: { ...state.semantics, [response.step]: response }, semanticsPending: null });
  },

  turnOffCulprits(): void {
    const found = state.bisect ? state.bisect.culprits : [];
    if (found.length === 0) return;
    actions.setOptions({ disabledPasses: [...new Set([...state.options.disabledPasses, ...found])] });
    void actions.run();
  },

  focusSource(line: number | null): void {
    if (state.focusLine !== line) set({ focusLine: line });
  },

  togglePass(name: string): void {
    const disabled = state.options.disabledPasses;
    actions.setOptions({
      disabledPasses: disabled.includes(name)
        ? disabled.filter(p => p !== name)
        : [...disabled, name],
    });
    void actions.run();
  },

  async run(): Promise<void> {
    if (state.status === 'compiling') return;
    const source = state.source;
    const options = state.options;
    set({ status: 'compiling' });

    const response = await ask({ kind: 'compile', source, options }) as CompileResponse;

    if (!response.ok && response.steps.length === 0) {
      set({
        status: 'failed',
        failure: { error: response.error ?? 'compile failed', errorPhase: response.errorPhase },
        playing: false,
        pane: 'source',
      });
      return;
    }

    const collapsed = collapsedByDefault(response);
    const next = { ...state, result: response, collapsed };
    const fallback = visibleSteps(next)[0];
    const suspect = response.ok ? null : firstSuspect(response);

    set({
      result: response,
      collapsed,
      failure: response.ok ? null : { error: response.error ?? 'compile failed', errorPhase: response.errorPhase },
      status: response.ok ? 'ready' : 'failed',
      ranSource: source,
      ranOptions: options,
      passPhases: phasesOf(response, state.passPhases),
      selected: suspect ?? (fallback ? fallback.index : 0),
      playing: false,
      pane: 'timeline',
      semantics: {},
      semanticsPending: null,
    });
  },
};
