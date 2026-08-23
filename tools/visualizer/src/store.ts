import { useSyncExternalStore } from 'react';
import { DEFAULT_OPTIONS } from './protocol.js';
import { EXAMPLES } from './examples/index.js';
import type { CompileOptions, CompileResponse, CompileStep, WorkerRequest, WorkerRequestDraft, WorkerResponse } from './protocol.js';

export type StageTab = 'ir' | 'graph' | 'why' | 'output' | 'result';

const REDUCED_MOTION = typeof matchMedia === 'function'
  && matchMedia('(prefers-reduced-motion: reduce)').matches;

export const SPEEDS = [0, 0.5, 1, 2] as const;
export type Status = 'starting' | 'idle' | 'compiling' | 'ready' | 'failed';
export type Pane = 'source' | 'timeline' | 'stage';

export type State = {
  source: string;
  exampleId: string;
  options: CompileOptions;
  globals: string[];
  status: Status;
  result: CompileResponse | null;
  selected: number;
  tab: StageTab;
  speed: number;
  playing: boolean;
  onlyChanged: boolean;
  pane: Pane;
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

let state: State = {
  source: EXAMPLES[0].source,
  exampleId: EXAMPLES[0].id,
  options: DEFAULT_OPTIONS,
  globals: [],
  status: 'starting',
  result: null,
  selected: 0,
  tab: 'ir',
  speed: REDUCED_MOTION ? 0 : 1,
  playing: false,
  onlyChanged: true,
  pane: 'source',
};

const listeners = new Set<() => void>();

function set(patch: Partial<State>): void {
  state = { ...state, ...patch };
  for (const listener of listeners) listener();
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

function nearestVisible(s: State, index: number): number {
  const steps = visibleSteps(s);
  if (steps.length === 0) return 0;
  let best = steps[0];
  for (const step of steps) {
    if (Math.abs(step.index - index) < Math.abs(best.index - index)) best = step;
  }
  return best.index;
}

export const actions = {
  async init(): Promise<void> {
    const response = await ask({ kind: 'init' }) as { globals: string[] };
    set({ globals: response.globals, status: 'idle' });
  },

  setSource(source: string): void {
    set({ source, exampleId: '' });
  },

  loadExample(id: string): void {
    const example = EXAMPLES.find(e => e.id === id);
    if (!example) return;
    set({ source: example.source, exampleId: id });
  },

  setOptions(patch: Partial<CompileOptions>): void {
    set({ options: { ...state.options, ...patch } });
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

  togglePass(name: string): void {
    const disabled = state.options.disabledPasses;
    set({
      options: {
        ...state.options,
        disabledPasses: disabled.includes(name)
          ? disabled.filter(p => p !== name)
          : [...disabled, name],
      },
    });
    void actions.run();
  },

  async run(): Promise<void> {
    if (state.status === 'compiling') return;
    set({ status: 'compiling' });
    const response = await ask({
      kind: 'compile',
      source: state.source,
      options: state.options,
    }) as CompileResponse;
    const next = { ...state, result: response };
    const first = visibleSteps(next)[0];
    set({
      result: response,
      status: response.ok ? 'ready' : 'failed',
      selected: first ? first.index : 0,
      playing: false,
      pane: response.ok ? 'timeline' : 'source',
    });
  },
};
