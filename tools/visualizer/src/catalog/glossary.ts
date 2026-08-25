import type { CompileStep } from '../protocol.js';

export type Mark = { glyph: string; label: string; tone: string };

const OUTCOME_MARKS: Record<CompileStep['outcome'], Mark> = {
  changed: { glyph: '●', label: 'the pass rewrote the IR', tone: 'changed' },
  unchanged: { glyph: '○', label: 'the pass ran and found nothing to do', tone: 'unchanged' },
  failed: { glyph: '✕', label: 'the pass threw before it finished', tone: 'failed' },
  unreported: { glyph: '◐', label: 'the pass did not say whether it changed anything', tone: 'unreported' },
};

const KIND_MARKS: Partial<Record<CompileStep['kind'], Mark>> = {
  input: { glyph: '▸', label: 'your model as tracing first recorded it', tone: 'input' },
  lowering: { glyph: '⇣', label: 'a change of language, not a pass', tone: 'lowering' },
};

export function markFor(step: Pick<CompileStep, 'kind' | 'outcome'>): Mark {
  return KIND_MARKS[step.kind] ?? OUTCOME_MARKS[step.outcome];
}

export const MARK_LEGEND: Mark[] = [
  KIND_MARKS.input as Mark,
  OUTCOME_MARKS.changed,
  OUTCOME_MARKS.unchanged,
  OUTCOME_MARKS.unreported,
  OUTCOME_MARKS.failed,
  KIND_MARKS.lowering as Mark,
];

export type Term = { term: string; meaning: string };

export const TERMS: Term[] = [
  {
    term: 'op',
    meaning: 'One unit of work in the compiler’s own language — a multiply, a broadcast, a matmul. Your two-line model becomes a few dozen of them.',
  },
  {
    term: 'IR',
    meaning: 'Intermediate representation: the program written in the compiler’s language instead of yours. Every pass reads one IR and writes it back.',
  },
  {
    term: 'pass',
    meaning: 'One transformation that reads the whole program and rewrites part of it. The compiler runs dozens in a fixed order; each is meant to do one thing.',
  },
  {
    term: 'graph IR',
    meaning: 'The first level: whole tensors flowing between ops, no loops yet. This is where fusion and algebraic cleanup happen.',
  },
  {
    term: 'tensor IR',
    meaning: 'The second level: every op has become an explicit loop nest over indices. Scheduling and memory planning work here.',
  },
  {
    term: 'low-level IR',
    meaning: 'The last level before text: flat addresses, registers, no tensor abstraction left. Almost the target language already.',
  },
  {
    term: 'lowering',
    meaning: 'Not a pass — a translation between two levels. Node counts jump here because one op becomes a whole loop nest.',
  },
  {
    term: 'fusion',
    meaning: 'Merging neighbouring ops into a single kernel so the intermediate values never touch memory. Usually the biggest single win.',
  },
  {
    term: 'kernel',
    meaning: 'The function the compiler finally emits — JavaScript, WebAssembly, CUDA C or WGSL — that actually computes your model.',
  },
  {
    term: 'eager',
    meaning: 'Running the model op by op with no compiler at all, the way a notebook would. The Result tab compares the compiled kernel against it.',
  },
  {
    term: 'trace',
    meaning: 'Running your model once with recording turned on, to capture the ops it performs. That recording is the input graph.',
  },
];

export type Shortcut = { keys: string; action: string };

export const SHORTCUTS: Shortcut[] = [
  { keys: 'Ctrl/Cmd + Enter', action: 'compile' },
  { keys: '↓ / j', action: 'next step' },
  { keys: '↑ / k', action: 'previous step' },
  { keys: 'Space', action: 'play or pause' },
  { keys: '?', action: 'open this guide' },
  { keys: 'Esc', action: 'close this guide' },
];

export const TAB_NOTES = [
  { id: 'ir', label: 'IR', meaning: 'The program before and after the selected step, as a diff.' },
  { id: 'graph', label: 'Graph', meaning: 'The same thing drawn as boxes and arrows, animated across the step.' },
  { id: 'why', label: 'Why', meaning: 'What the pass is for, and the decisions it recorded while running.' },
  { id: 'output', label: 'Output', meaning: 'The kernel source that falls out of the far end of the pipeline.' },
  { id: 'result', label: 'Result', meaning: 'That kernel actually executed, checked against the same model run eagerly.' },
] as const;

export type TabNote = (typeof TAB_NOTES)[number];

export type StageTab = TabNote['id'];
