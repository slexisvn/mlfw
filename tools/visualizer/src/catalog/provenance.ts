import type { CompileResponse, CompileStep } from '../protocol.js';

export type Presence = 'born' | 'alive' | 'died' | 'absent';

export type Provenance = {
  needle: string;
  marks: Map<number, Presence>;
  bornAt: number | null;
  diedAt: number | null;
  hits: number;
  insideLonger: number;
};

const ESCAPE = /[.*+?^${}()|[\]\\]/g;
const TOKEN_EDGE = '[A-Za-z0-9_%$]';

function matcher(needle: string, anchored: boolean): RegExp {
  const escaped = needle.replace(ESCAPE, '\\$&');
  return anchored ? new RegExp(`(?<!${TOKEN_EDGE})${escaped}(?!${TOKEN_EDGE})`) : new RegExp(escaped);
}

function presence(step: CompileStep, pattern: RegExp): Presence {
  const before = pattern.test(step.before.text);
  const after = pattern.test(step.after.text);
  if (before && after) return 'alive';
  if (after) return 'born';
  if (before) return 'died';
  return 'absent';
}

function countSteps(response: CompileResponse, pattern: RegExp): number {
  let seen = 0;
  for (const step of response.steps) {
    if (pattern.test(step.before.text) || pattern.test(step.after.text)) seen++;
  }
  return seen;
}

export function provenanceOf(response: CompileResponse, needle: string): Provenance | null {
  const trimmed = needle.trim();
  if (trimmed === '') return null;

  let anchored: RegExp;
  try {
    anchored = matcher(trimmed, true);
  } catch {
    return null;
  }

  const marks = new Map<number, Presence>();
  let bornAt: number | null = null;
  let diedAt: number | null = null;
  let hits = 0;

  for (const step of response.steps) {
    const mark = presence(step, anchored);
    marks.set(step.index, mark);
    if (mark === 'absent') continue;
    hits++;
    if (mark === 'born' && bornAt === null) bornAt = step.index;
    if (mark === 'died') diedAt = step.index;
  }

  return {
    needle: trimmed,
    marks,
    bornAt,
    diedAt,
    hits,
    insideLonger: hits > 0 ? 0 : countSteps(response, matcher(trimmed, false)),
  };
}

export const PRESENCE_MARKS: Record<Presence, { glyph: string; label: string }> = {
  born: { glyph: '+', label: 'this step is where it first appears' },
  alive: { glyph: '·', label: 'it is in the IR on both sides of this step' },
  died: { glyph: '×', label: 'this step is where it disappears' },
  absent: { glyph: '', label: 'not in the IR here' },
};
