import type { CompileResponse, CompileStep, IRLevelName } from '../protocol.js';

export type PhaseCost = {
  phase: string;
  level: IRLevelName;
  ms: number;
  runs: number;
  changed: number;
};

export type PassCost = {
  pass: string;
  phase: string;
  level: IRLevelName;
  ms: number;
  runs: number;
  changed: number;
};

export type Profile = {
  phases: PhaseCost[];
  passes: PassCost[];
  measuredMs: number;
  totalMs: number;
};

export type AnomalyKind = 'lied-changed' | 'lied-unchanged' | 'revisited' | 'invalid' | 'churned';

export type Anomaly = {
  kind: AnomalyKind;
  step: number;
  pass: string;
  detail: string;
};

const CHURN_RUNS = 4;

function key(step: CompileStep): string {
  return `${step.unit ?? ''}·${step.level}·${step.pass}`;
}

export function profileOf(response: CompileResponse): Profile {
  const phases = new Map<string, PhaseCost>();
  const passes = new Map<string, PassCost>();
  let measuredMs = 0;

  for (const step of response.steps) {
    if (step.kind !== 'pass') continue;
    measuredMs += step.durationMs;

    const phaseKey = `${step.phase}·${step.level}`;
    const phase = phases.get(phaseKey) ?? { phase: step.phase, level: step.level, ms: 0, runs: 0, changed: 0 };
    phase.ms += step.durationMs;
    phase.runs++;
    if (step.outcome === 'changed') phase.changed++;
    phases.set(phaseKey, phase);

    const passKey = `${step.pass}·${step.level}`;
    const pass = passes.get(passKey)
      ?? { pass: step.pass, phase: step.phase, level: step.level, ms: 0, runs: 0, changed: 0 };
    pass.ms += step.durationMs;
    pass.runs++;
    if (step.outcome === 'changed') pass.changed++;
    passes.set(passKey, pass);
  }

  const byCost = (a: { ms: number }, b: { ms: number }): number => b.ms - a.ms;

  return {
    phases: [...phases.values()].sort(byCost),
    passes: [...passes.values()].sort(byCost),
    measuredMs,
    totalMs: response.totalMs,
  };
}

function revisits(response: CompileResponse): Anomaly[] {
  const seenAt = new Map<string, CompileStep>();
  const found: Anomaly[] = [];
  let previous: string | null = null;

  for (const step of response.steps) {
    if (step.kind !== 'pass') continue;
    const state = `${step.unit ?? ''}·${step.level}·${step.after.text}`;
    const earlier = seenAt.get(state);

    if (earlier && previous !== state) {
      found.push({
        kind: 'revisited',
        step: step.index,
        pass: step.pass,
        detail: `the IR here is byte-identical to what ${earlier.pass} left at step ${earlier.index}, `
          + `and it was something else in between — the pipeline went round in a circle`,
      });
    } else if (!earlier) {
      seenAt.set(state, step);
    }

    previous = state;
  }

  return found;
}

function churn(response: CompileResponse): Anomaly[] {
  const runs = new Map<string, { step: CompileStep; total: number; changed: number }>();

  for (const step of response.steps) {
    if (step.kind !== 'pass') continue;
    const entry = runs.get(key(step)) ?? { step, total: 0, changed: 0 };
    entry.total++;
    if (step.outcome === 'changed') entry.changed++;
    runs.set(key(step), entry);
  }

  const found: Anomaly[] = [];
  for (const entry of runs.values()) {
    if (entry.total < CHURN_RUNS || entry.changed !== 0) continue;
    found.push({
      kind: 'churned',
      step: entry.step.index,
      pass: entry.step.pass,
      detail: `ran ${entry.total} times and never changed anything — the fixed-point loop is paying for it every round`,
    });
  }
  return found;
}

export function anomaliesOf(response: CompileResponse): Anomaly[] {
  const found: Anomaly[] = [];

  for (const step of response.steps) {
    if (step.kind !== 'pass') continue;

    if (step.verify && step.verify.introduced.length > 0) {
      found.push({
        kind: 'invalid',
        step: step.index,
        pass: step.pass,
        detail: step.verify.introduced[0],
      });
    }

    const identical = step.before.text === step.after.text;
    if (step.outcome === 'changed' && identical) {
      found.push({
        kind: 'lied-changed',
        step: step.index,
        pass: step.pass,
        detail: 'reported CHANGED, but the IR it produced is byte-identical to the IR it was given',
      });
    } else if (step.outcome === 'unchanged' && !identical) {
      found.push({
        kind: 'lied-unchanged',
        step: step.index,
        pass: step.pass,
        detail: 'reported UNCHANGED, but the IR is different afterwards — later passes are told nothing moved',
      });
    }
  }

  return [...found, ...revisits(response), ...churn(response)].sort((a, b) => a.step - b.step);
}

export const ANOMALY_NOTES: Record<AnomalyKind, { label: string; why: string }> = {
  invalid: {
    label: 'invalid IR',
    why: 'the verifier accepted the IR going into this pass and rejects what came out',
  },
  'lied-changed': {
    label: 'false CHANGED',
    why: 'a pass that claims it changed something re-runs its fixed-point group and invalidates analyses for nothing',
  },
  'lied-unchanged': {
    label: 'false UNCHANGED',
    why: 'the dangerous one: stale analyses are kept and the fixed-point loop stops early on a graph that is still moving',
  },
  revisited: {
    label: 'went in a circle',
    why: 'two passes are undoing each other, so the pipeline is burning iterations to end where it started',
  },
  churned: {
    label: 'never fires',
    why: 'the pass costs time on every iteration of its group and has never had anything to do on this program',
  },
};
