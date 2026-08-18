export type OptCandidate = Readonly<{
  name: string;
  optimization?: Readonly<Record<string, unknown>>;
  scheduling?: Readonly<Record<string, unknown>>;
}>;

export type CandidateMeasurement = Readonly<{
  name: string;
  ms: number;
  correct: boolean;
  error?: string;
}>;

export type GateDecision = Readonly<{
  winner: string;
  baselineMs: number;
  winnerMs: number;
  gain: number;
  measurements: readonly CandidateMeasurement[];
}>;

export const BASELINE = 'baseline';

export const DEFAULT_MIN_GAIN = 1.05;

export type GateTarget = {
  isGPU?(): boolean;
  supportsTensorCore?: boolean;
  supportsBlockedLayout?: boolean;
  preferredConvLayout?: string | null;
};

export function optimizationCandidates(target: GateTarget | null): OptCandidate[] {
  if (!target) return [];
  const candidates: OptCandidate[] = [];
  const gpu = typeof target.isGPU === 'function' && target.isGPU();

  if (target.supportsBlockedLayout || target.preferredConvLayout) {
    candidates.push({ name: 'layout', optimization: { layout: true } });
  }
  if (gpu && target.supportsTensorCore) {
    candidates.push({ name: 'tensorize', optimization: { tensorize: true } });
  }
  if (gpu && target.supportsTensorCore && (target.supportsBlockedLayout || target.preferredConvLayout)) {
    candidates.push({ name: 'layout+tensorize', optimization: { layout: true, tensorize: true } });
  }
  return candidates;
}

export function selectWinner(
  measurements: readonly CandidateMeasurement[],
  minGain = DEFAULT_MIN_GAIN,
): GateDecision {
  const baseline = measurements.find(m => m.name === BASELINE);
  if (!baseline || !baseline.correct) {
    throw new Error(`optimization gate: the ${BASELINE} configuration must be measured and correct before candidates can be compared`);
  }

  let best = baseline;
  for (const m of measurements) {
    if (m.name === BASELINE || !m.correct || !(m.ms > 0)) continue;
    if (m.ms < best.ms) best = m;
  }

  const gain = best.ms > 0 ? baseline.ms / best.ms : 1;
  const winner = best.name !== BASELINE && gain >= minGain ? best : baseline;
  return {
    winner: winner.name,
    baselineMs: baseline.ms,
    winnerMs: winner.ms,
    gain: winner.name === BASELINE ? 1 : gain,
    measurements,
  };
}

export function candidateByName(candidates: readonly OptCandidate[], name: string): OptCandidate | null {
  if (name === BASELINE) return null;
  return candidates.find(c => c.name === name) ?? null;
}

export function gateCacheKey(signature: string, targetName: string, candidates: readonly OptCandidate[]): string {
  return `${targetName}|${candidates.map(c => c.name).join(',')}|${signature}`;
}

export function graphSignature(opNames: Iterable<string>, inputShapes: Iterable<readonly number[]>): string {
  const ops: string[] = [];
  for (const n of opNames) ops.push(n);
  const shapes: string[] = [];
  for (const s of inputShapes) shapes.push(s.join('x'));
  return `${ops.join(',')}#${shapes.join(';')}`;
}
