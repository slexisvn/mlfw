import { agree, plural } from './naming.js';
import type { CompileOptions, CompileResponse, IRLevelName, PassOutcome } from '../protocol.js';

export type LedgerEntry = {
  key: string;
  pass: string;
  level: IRLevelName;
  phase: string;
  run: number;
  outcome: PassOutcome;
  before: number;
  after: number;
};

export type Ledger = {
  format: 'mlfw-pass-ledger';
  savedAt: string;
  source: string;
  options: CompileOptions;
  totalMs: number;
  entries: LedgerEntry[];
};

export type LedgerChange = { key: string; entry: LedgerEntry; was: LedgerEntry };

export type LedgerDiff = {
  matched: number;
  changed: LedgerChange[];
  onlyOld: LedgerEntry[];
  onlyNew: LedgerEntry[];
};

export function ledgerOf(response: CompileResponse, source: string, options: CompileOptions): Ledger {
  const runs = new Map<string, number>();
  const entries: LedgerEntry[] = [];

  for (const step of response.steps) {
    if (step.kind !== 'pass') continue;
    const base = `${step.pass}·${step.level}`;
    const run = (runs.get(base) ?? 0) + 1;
    runs.set(base, run);
    entries.push({
      key: `${base}·${run}`,
      pass: step.pass,
      level: step.level,
      phase: step.phase,
      run,
      outcome: step.outcome,
      before: step.before.ops,
      after: step.after.ops,
    });
  }

  return {
    format: 'mlfw-pass-ledger',
    savedAt: new Date().toISOString(),
    source,
    options,
    totalMs: response.totalMs,
    entries,
  };
}

export function parseLedger(text: string): Ledger | null {
  try {
    const parsed = JSON.parse(text) as Partial<Ledger>;
    if (parsed.format !== 'mlfw-pass-ledger' || !Array.isArray(parsed.entries)) return null;
    return parsed as Ledger;
  } catch {
    return null;
  }
}

function differs(a: LedgerEntry, b: LedgerEntry): boolean {
  return a.before !== b.before || a.after !== b.after || a.outcome !== b.outcome;
}

export function diffLedgers(was: Ledger, now: Ledger): LedgerDiff {
  const old = new Map(was.entries.map(entry => [entry.key, entry]));
  const changed: LedgerChange[] = [];
  const onlyNew: LedgerEntry[] = [];
  let matched = 0;

  for (const entry of now.entries) {
    const before = old.get(entry.key);
    if (!before) {
      onlyNew.push(entry);
      continue;
    }
    old.delete(entry.key);
    matched++;
    if (differs(before, entry)) changed.push({ key: entry.key, entry, was: before });
  }

  return { matched, changed, onlyOld: [...old.values()], onlyNew };
}

export function summarize(diff: LedgerDiff): string {
  if (diff.changed.length === 0 && diff.onlyOld.length === 0 && diff.onlyNew.length === 0) {
    return `Identical: all ${diff.matched} pass runs did the same thing to the same number of ops.`;
  }
  const parts: string[] = [];
  if (diff.changed.length > 0) parts.push(`${plural(diff.changed.length, 'pass run')} ${agree(diff.changed.length, 'behaves')} differently`);
  if (diff.onlyOld.length > 0) parts.push(`${plural(diff.onlyOld.length, 'run')} no longer ${agree(diff.onlyOld.length, 'happens')}`);
  if (diff.onlyNew.length > 0) parts.push(`${plural(diff.onlyNew.length, 'run')} ${agree(diff.onlyNew.length, 'is')} new`);
  return `${parts.join(', ')} — out of ${diff.matched} that line up.`;
}
