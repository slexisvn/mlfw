import { useMemo } from 'react';
import type { TuningRound } from '../protocol.js';

const CHART_HEIGHT = 64;
const DOT_RADIUS = 2.2;
const MIN_SPREAD = 1e-9;

type Block = {
  key: string;
  func: string;
  blockName: string;
  rounds: TuningRound[];
  best: number | null;
  bestSketch: string | null;
  worst: number | null;
  tried: number;
  sketches: Set<string>;
  measured: boolean;
};

function groupByBlock(rounds: readonly TuningRound[]): Block[] {
  const blocks = new Map<string, Block>();

  for (const round of rounds) {
    const key = `${round.func}::${round.blockName}`;
    let block = blocks.get(key);
    if (!block) {
      block = {
        key, func: round.func, blockName: round.blockName, rounds: [],
        best: null, bestSketch: null, worst: null, tried: 0,
        sketches: new Set(), measured: false,
      };
      blocks.set(key, block);
    }

    block.rounds.push(round);
    block.tried += round.scores.length;
    block.measured = block.measured || round.measured;
    for (const candidate of round.scores) {
      block.sketches.add(candidate.sketch);
      block.worst = block.worst === null ? candidate.score : Math.min(block.worst, candidate.score);
      block.best = block.best === null ? candidate.score : Math.max(block.best, candidate.score);
    }
    if (round.bestSketch) block.bestSketch = round.bestSketch;
  }

  return [...blocks.values()];
}

export function TuningPanel({ rounds }: { rounds: readonly TuningRound[] }) {
  const blocks = useMemo(() => groupByBlock(rounds), [rounds]);

  if (blocks.length === 0) {
    return (
      <div className="pane-empty">
        No search ran. Turn on <strong>tune</strong> in the options and the scheduler stops guessing from
        rules and starts searching instead — every schedule it tries shows up here with the score that
        decided its fate.
      </div>
    );
  }

  return (
    <div className="tuning">
      <section className="memory-verdict">
        <h2>{blocks.length} block{blocks.length === 1 ? '' : 's'} searched</h2>
        <p>
          A sketch is a shape of schedule with holes in it — tile sizes, unroll factors, which loop binds to
          which thread. Filling the holes gives a candidate, and every candidate below was scored without
          being run{blocks.some(b => b.measured) ? ', then the best few were actually timed' : ''}. The
          scores are a cost model’s opinion, not a measurement: what this pane shows is how wide the space
          was and how far apart its opinions were.
        </p>
      </section>

      {blocks.map(block => <BlockChart key={block.key} block={block} />)}
    </div>
  );
}

function spreadNote(best: number, worst: number, spread: number): string {
  if (spread <= MIN_SPREAD * 10) return 'every candidate scored the same — the model cannot tell them apart';
  if (worst <= 0) return `${spread.toFixed(4)} between the best and the worst it kept`;
  return `the best scores ${(best / worst).toFixed(2)}× the worst it kept`;
}

function BlockChart({ block }: { block: Block }) {
  const best = block.best ?? 0;
  const worst = block.worst ?? 0;
  const spread = Math.max(best - worst, MIN_SPREAD);
  const width = Math.max(block.rounds.length, 1);
  const sketches = [...block.sketches];

  return (
    <section className="memory-block">
      <h3 className="tuning-head">
        <code>{block.blockName}</code>
        <span>
          {block.tried} candidate{block.tried === 1 ? '' : 's'} over {block.rounds.length} round
          {block.rounds.length === 1 ? '' : 's'} · {sketches.length} sketch
          {sketches.length === 1 ? '' : 'es'}
        </span>
      </h3>

      <div
        className="tuning-chart"
        role="img"
        aria-label={
          `${block.tried} candidates scored between ${worst.toFixed(3)} and ${best.toFixed(3)}`
          + ` across ${block.rounds.length} rounds`
        }
      >
        {block.rounds.map((round, i) => round.scores.map((candidate, j) => (
          <span
            key={`${round.round}-${j}`}
            className={candidate.score === best ? 'candidate best' : 'candidate'}
            style={{
              left: `${((i + (j + 0.5) / round.scores.length) / width) * 100}%`,
              bottom: `${((candidate.score - worst) / spread) * 100}%`,
            }}
            title={`${candidate.sketch} · score ${candidate.score.toFixed(4)}`}
          />
        )))}
      </div>

      <p className="memory-legend">
        best {best.toFixed(4)}
        {block.bestSketch && <span className="tuning-winner">{block.bestSketch}</span>}
        worst kept {worst.toFixed(4)}
        <span className="tuning-flat">{spreadNote(best, worst, spread)}</span>
      </p>
    </section>
  );
}
