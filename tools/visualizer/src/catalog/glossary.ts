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
  { id: 'memory', label: 'Memory', meaning: 'Which buffer is alive when, and how many of them share one slot.' },
  { id: 'tuning', label: 'Tuning', meaning: 'Every schedule the search tried, and the score that picked one.' },
  { id: 'compare', label: 'Compare', meaning: 'This run against one you pinned earlier, measure by measure.' },
  { id: 'output', label: 'Output', meaning: 'The kernel source that falls out of the far end of the pipeline.' },
  { id: 'result', label: 'Result', meaning: 'That kernel actually executed, checked against the same model run eagerly.' },
] as const;

export type TabNote = (typeof TAB_NOTES)[number];

export type StageTab = TabNote['id'];
