import { useMemo } from 'react';
import { knobNote, sketchNote } from '../catalog/tuning.js';
import type { TuningCandidate, TuningParams, TuningRound } from '../protocol.js';

const TIE_EPSILON = 1e-9;
const PLOT_FLOOR = 10;
const PLOT_CEILING = 90;

type Knob = { name: string; values: string[] };

type Block = {
  key: string;
  func: string;
  blockName: string;
  candidates: TuningCandidate[];
  roundEnds: number[];
  sketches: string[];
  knobs: Knob[];
  best: number;
  worst: number;
  spread: number;
  bestSketch: string | null;
  bestParams: TuningParams | null;
  bestMedianMs: number | null;
  measured: boolean;
};

function paramText(value: number | number[]): string {
  return Array.isArray(value) ? value.join('×') : String(value);
}

function paramsText(params: TuningParams | null): string {
  if (!params) return '';
  return Object.entries(params).map(([name, value]) => `${name} ${paramText(value)}`).join(' · ');
}

function collectKnobs(candidates: readonly TuningCandidate[]): Knob[] {
  const seen = new Map<string, Set<string>>();
  for (const candidate of candidates) {
    for (const [name, value] of Object.entries(candidate.params)) {
      let values = seen.get(name);
      if (!values) { values = new Set(); seen.set(name, values); }
      values.add(paramText(value));
    }
  }

  const knobs: Knob[] = [];
  for (const [name, values] of seen) {
    if (values.size < 2) continue;
    knobs.push({ name, values: [...values].sort((a, b) => (Number(a) - Number(b)) || a.localeCompare(b)) });
  }
  return knobs;
}

function groupByBlock(rounds: readonly TuningRound[]): Block[] {
  const blocks = new Map<string, Block>();

  for (const round of rounds) {
    const key = `${round.func}::${round.blockName}`;
    let block = blocks.get(key);
    if (!block) {
      block = {
        key, func: round.func, blockName: round.blockName,
        candidates: [], roundEnds: [], sketches: [], knobs: [],
        best: -Infinity, worst: Infinity, spread: 0,
        bestSketch: null, bestParams: null, bestMedianMs: null, measured: false,
      };
      blocks.set(key, block);
    }

    block.measured = block.measured || round.measured;
    for (const candidate of round.scores) {
      block.candidates.push(candidate);
      block.worst = Math.min(block.worst, candidate.score);
      block.best = Math.max(block.best, candidate.score);
    }
    block.roundEnds.push(block.candidates.length);
    if (round.bestSketch) {
      block.bestSketch = round.bestSketch;
      block.bestParams = round.bestParams;
    }
    if (round.bestMedianMs !== null) block.bestMedianMs = round.bestMedianMs;
  }

  for (const block of blocks.values()) {
    block.sketches = [...new Set(block.candidates.map(candidate => candidate.sketch))];
    block.knobs = collectKnobs(block.candidates);
    block.spread = block.best - block.worst;
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

  const separated = blocks.filter(block => block.spread > TIE_EPSILON).sort((a, b) => b.spread - a.spread);
  const tied = blocks.filter(block => block.spread <= TIE_EPSILON);
  const tried = blocks.reduce((sum, block) => sum + block.candidates.length, 0);
  const funcs = new Set(blocks.map(block => block.func));
  const measured = blocks.some(block => block.measured);

  return (
    <div className="tuning">
      <section className="memory-verdict">
        <h2>
          {separated.length === 0
            ? `${tried} schedules tried, none of them the model could tell apart`
            : `${tried} schedules tried, and the score separated ${separated.length} of ${blocks.length} blocks`}
        </h2>
        <p>
          Every block below is one loop nest, and the scheduler has to decide how to run it: how many threads
          take it, how wide a vector is, how far a reduction is cut. A <strong>sketch</strong> is that decision
          with the numbers left as holes; filling the holes gives a <strong>candidate</strong>. The search
          filled them {tried} time{tried === 1 ? '' : 's'} and handed each result to a cost model, which
          returns a score where <strong>higher means it expects that schedule to be faster</strong>.{' '}
          {measured
            ? 'The best few were then compiled and timed for real.'
            : 'Nothing here was run — these are opinions, not measurements.'}
          {' '}A score only means something inside its own block: the same number in two charts is a
          coincidence, not a comparison.
        </p>
      </section>

      {separated.length > 0 && (
        <section className="tuning-key">
          <span><span className="key best" /> the schedule that won</span>
          <span><span className="key other" /> a candidate that scored lower</span>
          <span>higher in the box = better score</span>
          <span>left to right = the order they were tried</span>
          <span>hover a dot for its sketch and numbers</span>
        </section>
      )}

      {separated.map(block => (
        <BlockChart key={block.key} block={block} showFunc={funcs.size > 1} />
      ))}

      {tied.length > 0 && <TiedBlocks blocks={tied} showFunc={funcs.size > 1} />}
    </div>
  );
}

function gapNote(block: Block): string {
  if (block.worst > 0) return `the winner scores ${(block.best / block.worst).toFixed(2)}× the worst candidate`;
  return `${block.spread.toFixed(4)} of score between the winner and the worst candidate`;
}

function BlockChart({ block, showFunc }: { block: Block; showFunc: boolean }) {
  const count = block.candidates.length;
  const rounds = block.roundEnds.length;
  const height = PLOT_CEILING - PLOT_FLOOR;

  return (
    <section className="memory-block">
      <h3 className="tuning-head">
        <code>{block.blockName}</code>
        <span>
          {showFunc && `${block.func} · `}
          {count} candidate{count === 1 ? '' : 's'} · {block.sketches.length} sketch
          {block.sketches.length === 1 ? '' : 'es'}
          {rounds > 1 && ` · ${rounds} rounds`}
        </span>
      </h3>

      <div className="tuning-plot">
        <div className="tuning-scale">
          <span>{block.best.toFixed(3)}</span>
          <span className="tuning-scale-dir">better ↑</span>
          <span>{block.worst.toFixed(3)}</span>
        </div>

        <div
          className="tuning-chart"
          role="img"
          aria-label={
            `${count} candidates for ${block.blockName}, scored between ${block.worst.toFixed(3)}`
            + ` and ${block.best.toFixed(3)}`
          }
        >
          {block.roundEnds.slice(0, -1).map(end => (
            <span key={end} className="tuning-round" style={{ left: `${(end / count) * 100}%` }} />
          ))}
          {block.candidates.map((candidate, i) => (
            <span
              key={i}
              className={candidate.score === block.best ? 'candidate best' : 'candidate'}
              style={{
                left: `${((i + 0.5) / count) * 100}%`,
                bottom: `${PLOT_FLOOR + ((candidate.score - block.worst) / block.spread) * height}%`,
              }}
              title={
                `${candidate.sketch}${paramsText(candidate.params) ? ` · ${paramsText(candidate.params)}` : ''}`
                + ` · score ${candidate.score.toFixed(4)}`
              }
            />
          ))}
        </div>
      </div>

      <p className="tuning-axis">
        candidate 1 on the left, candidate {count} on the right
        {rounds > 1 && ', and the faint lines are where one search round ended and the next began'}
      </p>

      <p className="memory-legend">
        <span className="key best" />
        won with <code>{block.bestSketch ?? block.sketches[0]}</code>
        {paramsText(block.bestParams) && <span className="tuning-winner">{paramsText(block.bestParams)}</span>}
        {block.bestMedianMs !== null && <span>timed at {block.bestMedianMs.toFixed(3)} ms</span>}
        <span className="tuning-flat">{gapNote(block)}</span>
      </p>

      {block.bestSketch && sketchNote(block.bestSketch) && (
        <p className="tuning-note">That sketch {sketchNote(block.bestSketch)}.</p>
      )}

      {block.knobs.map(knob => (
        <p className="tuning-note" key={knob.name}>
          <code>{knob.name}</code>
          {knobNote(knob.name) ? ` (${knobNote(knob.name)})` : ''} was the hole the search filled — it tried{' '}
          {knob.values.join(', ')}.
        </p>
      ))}
    </section>
  );
}

function TiedBlocks({ blocks, showFunc }: { blocks: readonly Block[]; showFunc: boolean }) {
  return (
    <section className="memory-block">
      <h3>
        {blocks.length} block{blocks.length === 1 ? '' : 's'} the model could not separate
        <span>every candidate scored the same, so searching them changed nothing</span>
      </h3>

      <div className="tuning-tied">
        {blocks.map(block => (
          <div className="tuning-tied-row" key={block.key}>
            <code>{showFunc ? `${block.func}::${block.blockName}` : block.blockName}</code>
            <span>{block.candidates.length} candidates</span>
            <span>all scored {block.best.toFixed(4)}</span>
            <span className="tuning-winner">
              {block.bestSketch ?? block.sketches[0]}
              {paramsText(block.bestParams) ? ` · ${paramsText(block.bestParams)}` : ''}
            </span>
          </div>
        ))}
      </div>

      <p className="tuning-note">
        A tie means the search turned knobs this cost model does not look at, so the first candidate stands
        and the time spent searching bought nothing.
      </p>
    </section>
  );
}
