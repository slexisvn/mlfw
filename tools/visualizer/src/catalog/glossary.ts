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
  primitive: { glyph: '↳', label: 'one schedule primitive inside the pass above', tone: 'primitive' },
};

export function markFor(step: Pick<CompileStep, 'kind' | 'outcome'>): Mark {
  return KIND_MARKS[step.kind] ?? OUTCOME_MARKS[step.outcome];
}

export const TAB_NOTES = [
  { id: 'ir', label: 'IR', meaning: 'The program before and after the selected step, as a diff.' },
  { id: 'graph', label: 'Graph', meaning: 'The same thing drawn as boxes and arrows, animated across the step.' },
  { id: 'why', label: 'Why', meaning: 'What the pass is for, and the decisions it recorded while running.' },
  { id: 'semantics', label: 'Semantics', meaning: 'Run the loop nest before and after the pass and compare every store — proof, not inspection.' },
  { id: 'memory', label: 'Memory', meaning: 'Which buffer is alive when, and how many of them share one slot.' },
  { id: 'tuning', label: 'Tuning', meaning: 'Every schedule the search tried, and the score that picked one.' },
  { id: 'compare', label: 'Compare', meaning: 'This run against one you pinned earlier, measure by measure.' },
  { id: 'bisect', label: 'Bisect', meaning: 'Turn passes off in a search until the smallest set that is breaking this compile is named.' },
  { id: 'profile', label: 'Profile', meaning: 'Where the compile time went, and everything the pipeline did that it should not have.' },
  { id: 'trace', label: 'Trace', meaning: 'Every event the compiler emitted, unfiltered and searchable.' },
  { id: 'output', label: 'Output', meaning: 'The kernel source that falls out of the far end of the pipeline.' },
  { id: 'result', label: 'Result', meaning: 'That kernel actually executed, checked against the same model run eagerly.' },
  { id: 'health', label: 'Health', meaning: 'What every layer and gradient actually contained — ranges, norms, and anything that is not a number.' },
] as const;

export type TabNote = (typeof TAB_NOTES)[number];

export type StageTab = TabNote['id'];
