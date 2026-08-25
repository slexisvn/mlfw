import { useSyncExternalStore } from 'react';
import { DEFAULT_OPTIONS } from './protocol.js';
import { EXAMPLES } from './examples/index.js';
import { readSession, shareUrl, writeSession } from './session.js';
import type { CompileOptions, CompileResponse, CompileStep, WorkerRequest, WorkerRequestDraft, WorkerResponse } from './protocol.js';
import type { StageTab } from './catalog/glossary.js';

const REDUCED_MOTION = typeof matchMedia === 'function'
  && matchMedia('(prefers-reduced-motion: reduce)').matches;

export const SPEEDS = [0, 0.5, 1, 2] as const;
export type Status = 'starting' | 'idle' | 'compiling' | 'ready' | 'failed';
export type Pane = 'source' | 'timeline' | 'stage';

export type Failure = { error: string; errorPhase: string | null };

export type State = {
  source: string;
  exampleId: string;
  options: CompileOptions;
  globals: string[];
  status: Status;
  result: CompileResponse | null;
  failure: Failure | null;
  ranSource: string | null;
  ranOptions: CompileOptions | null;
  passPhases: Record<string, string>;
  selected: number;
  tab: StageTab;
  speed: number;
  playing: boolean;
  onlyChanged: boolean;
  pane: Pane;
  focusLine: number | null;
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
  failure: null,
  ranSource: null,
  ranOptions: null,
  passPhases: {},
  selected: 0,
  tab: 'ir',
  speed: REDUCED_MOTION ? 0 : 1,
  playing: false,
  onlyChanged: true,
  pane: 'source',
  focusLine: null,
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

const visibleCache = new WeakMap<State, CompileStep[]>();

export function visibleSteps(s: State): CompileStep[] {
  const cached = visibleCache.get(s);
  if (cached) return cached;

  const computed = !s.result
    ? []
    : s.onlyChanged
      ? s.result.steps.filter(step => step.kind !== 'pass' || step.outcome !== 'unchanged')
      : s.result.steps;
  const steps = computed.length > 0 || !s.result ? computed : s.result.steps;

  visibleCache.set(s, steps);
  return steps;
}

export function changedCount(s: State): number {
  if (!s.result) return 0;
  return s.result.steps.filter(step => step.kind !== 'pass' || step.outcome !== 'unchanged').length;
}

export function isStale(s: State): boolean {
  if (!s.result || s.ranOptions === null) return false;
  if (s.ranSource !== s.source) return true;
  const keys = Object.keys(s.options) as (keyof CompileOptions)[];
  return keys.some(key => JSON.stringify(s.options[key]) !== JSON.stringify((s.ranOptions as CompileOptions)[key]));
}

const sourceLineCache = new WeakMap<CompileResponse, Map<number, number>>();
const NO_LINES: Map<number, number> = new Map();

export function sourceLines(s: State): Map<number, number> {
  if (!s.result) return NO_LINES;
  const cached = sourceLineCache.get(s.result);
  if (cached) return cached;
  const map = new Map(s.result.sourceLinks);
  sourceLineCache.set(s.result, map);
  return map;
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

  toggleOnlyChanged(): void {
    const next = { ...state, onlyChanged: !state.onlyChanged };
    set({ onlyChanged: next.onlyChanged, selected: nearestVisible(next, state.selected), playing: false });
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

  share(): string {
    return shareUrl({ source: state.source, exampleId: state.exampleId, options: state.options });
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

    const next = { ...state, result: response };
    const first = visibleSteps(next)[0];
    set({
      result: response,
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
