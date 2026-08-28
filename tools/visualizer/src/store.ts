import { useSyncExternalStore } from 'react';
import { DEFAULT_OPTIONS } from './protocol.js';
import { EXAMPLES } from './examples/index.js';
import { readSession, writeSession } from './session.js';
import type { CompileOptions, CompileResponse, CompileStep, WorkerRequest, WorkerRequestDraft, WorkerResponse } from './protocol.js';
import { lessonById } from './catalog/lessons.js';
import type { Lesson } from './catalog/lessons.js';
import type { StageTab } from './catalog/glossary.js';

const REDUCED_MOTION = typeof matchMedia === 'function'
  && matchMedia('(prefers-reduced-motion: reduce)').matches;

const RUNS: ReadonlySet<string> = new Set(['pass', 'primitive']);

export const SPEEDS = [0, 0.5, 1, 2] as const;
export type Status = 'starting' | 'idle' | 'compiling' | 'ready' | 'failed';
export type Pane = 'source' | 'timeline' | 'stage';

export type Failure = { error: string; errorPhase: string | null };
export type Baseline = { response: CompileResponse; options: CompileOptions; source: string };
export type LessonProgress = { id: string; at: number; picked: number | null };

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
  lesson: LessonProgress | null;
};

const worker = new Worker(new URL('./worker/compile.worker.ts', import.meta.url), { type: 'module' });

let nextRequestId = 1;
const pending = new Map<number, (response: WorkerResponse) => void>();

worker.onmessage = (message: MessageEvent<WorkerResponse>) => {
  const resolve = pending.get(message.data.id);
  if (!resolve) return;
  pending.delete(message.data.id);
  resolve(message.data);
};

function ask(request: WorkerRequestDraft): Promise<WorkerResponse> {
  const id = nextRequestId++;
  return new Promise(resolve => {
    pending.set(id, resolve);
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
  lesson: null,
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

function collapsedByDefault(response: CompileResponse): Set<number> {
  return new Set(family(response).childrenOf.keys());
}

const visibleCache = new WeakMap<State, CompileStep[]>();

export function visibleSteps(s: State): CompileStep[] {
  const cached = visibleCache.get(s);
  if (cached) return cached;

  const computed = !s.result ? [] : s.result.steps.filter(step => shown(s, step));
  const steps = computed.length > 0 || !s.result ? computed : s.result.steps;

  visibleCache.set(s, steps);
  return steps;
}

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

export type DisabledPass = { name: string; phase: string };

const disabledCache = new WeakMap<State, DisabledPass[]>();

export function disabledPasses(s: State): DisabledPass[] {
  const cached = disabledCache.get(s);
  if (cached) return cached;
  const entries = s.options.disabledPasses.map(name => ({ name, phase: s.passPhases[name] ?? 'turned off' }));
  disabledCache.set(s, entries);
  return entries;
}

function nearestVisible(s: State, index: number): number {
  const steps = visibleSteps(s);
  if (steps.length === 0) return 0;
  let best = steps[0];
  for (const step of steps) {
    if (Math.abs(step.index - index) < Math.abs(best.index - index)) best = step;
  }
  return best.index;
}

function phasesOf(response: CompileResponse, previous: Record<string, string>): Record<string, string> {
  const merged = { ...previous };
  for (const step of response.steps) {
    if (step.kind === 'pass') merged[step.pass] = step.phase;
  }
  return merged;
}

function focusStep(pass: string, tab: StageTab): void {
  const result = state.result;
  if (!result) return;
  const step = result.steps.find(candidate => candidate.pass === pass);
  set({ tab, pane: 'stage' });
  if (!step) return;
  if (family(result).childrenOf.has(step.index)) {
    const collapsed = new Set(state.collapsed);
    collapsed.delete(step.index);
    set({ collapsed });
  }
  actions.select(step.index);
}

async function applyBeat(lesson: Lesson, at: number): Promise<void> {
  const beat = lesson.beats[at];
  const patch: Partial<State> = { lesson: { id: lesson.id, at, picked: null } };
  if (beat.source !== undefined) {
    patch.source = beat.source;
    patch.exampleId = '';
  }
  if (beat.options) patch.options = { ...state.options, ...beat.options };
  set(patch);
  persist();

  if (beat.run) await actions.run();
  if (beat.focus) focusStep(beat.focus.pass, beat.focus.tab);
  else if (beat.tab) set({ tab: beat.tab, pane: 'stage' });
}

export const actions = {
  async init(): Promise<void> {
    const response = await ask({ kind: 'init' }) as { globals: string[] };
    set({ globals: response.globals, status: 'idle' });
  },

  setSource(source: string): void {
    set({ source, exampleId: '' });
    persist();
  },

  loadExample(id: string): void {
    const example = EXAMPLES.find(e => e.id === id);
    if (!example) return;
    set({ source: example.source, exampleId: id });
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

  async startLesson(id: string): Promise<void> {
    const lesson = lessonById(id);
    if (!lesson) return;
    await applyBeat(lesson, 0);
  },

  endLesson(): void {
    set({ lesson: null });
  },

  async gotoBeat(at: number): Promise<void> {
    const progress = state.lesson;
    const lesson = progress ? lessonById(progress.id) : null;
    if (!lesson) return;
    await applyBeat(lesson, Math.max(0, Math.min(lesson.beats.length - 1, at)));
  },

  pick(choice: number): void {
    if (!state.lesson) return;
    set({ lesson: { ...state.lesson, picked: choice } });
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

  focusSource(line: number | null): void {
    set({ focusLine: line });
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

    if (!response.ok) {
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
    const first = visibleSteps(next)[0];
    set({
      result: response,
      collapsed,
      failure: null,
      status: 'ready',
      ranSource: source,
      ranOptions: options,
      passPhases: phasesOf(response, state.passPhases),
      selected: first ? first.index : 0,
      playing: false,
      pane: 'timeline',
    });
  },
};
