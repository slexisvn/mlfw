import { useState } from 'react';
import { actions, useStore } from '../store.js';
import { DEFAULT_TOLERANCE, TOLERANCES } from '../catalog/diagnostics.js';
import { passLabel, plural } from '../catalog/naming.js';
import type { BisectProbe, BisectResponse, CompileResponse } from '../protocol.js';

function diffText(probe: BisectProbe): string {
  if (!probe.ok) return 'threw';
  if (probe.error !== null) return 'run threw';
  if (!probe.ran) return 'compiled';
  if (probe.diff === null) return 'no comparison';
  return probe.diff === 0 ? 'exact' : probe.diff.toExponential(1);
}

function whatBroke(result: BisectResponse): string {
  if (result.mode === 'compile') return 'this compile threw';
  if (result.mode === 'numeric') return 'this compile answered differently from eager';
  return 'this compile is fine';
}

type Symptom = { headline: string; blurb: string; broken: boolean };

function symptomOf(result: CompileResponse | null, tolerance: number): Symptom {
  if (!result) {
    return {
      headline: 'nothing to bisect yet',
      blurb: 'Press Run first. The search needs a compile to reproduce before it can start turning passes off.',
      broken: false,
    };
  }

  if (!result.ok) {
    return {
      headline: 'this compile threw — worth bisecting',
      blurb: `The pipeline failed with “${result.error ?? 'an error'}”. The search turns passes off in groups until `
        + 'the smallest set that has to be off for the compile to survive is named.',
      broken: true,
    };
  }

  const run = result.run;
  if (run.error !== null) {
    return {
      headline: 'the compiled kernel threw — worth bisecting',
      blurb: `Running it failed with “${run.error}”. The oracle is whether the compiled answer comes back at all.`,
      broken: true,
    };
  }

  if (run.ran && run.maxAbsDiff !== null && run.maxAbsDiff > tolerance) {
    return {
      headline: 'this compile answers differently from eager — worth bisecting',
      blurb: `The worst element is off by ${run.maxAbsDiff.toExponential(1)}, past the ${tolerance.toExponential(0)} `
        + 'tolerance below. The search finds the pass responsible.',
      broken: true,
    };
  }

  if (run.ran && run.maxAbsDiff !== null) {
    return {
      headline: 'nothing is broken to bisect',
      blurb: `This compile already matches eager to ${run.maxAbsDiff.toExponential(1)}, inside the `
        + `${tolerance.toExponential(0)} tolerance below. Tighten the tolerance, or change something that breaks it, `
        + 'and the search has an oracle to work against.',
      broken: false,
    };
  }

  return {
    headline: 'no oracle to bisect against',
    blurb: `The compiled kernel did not run${run.skipped ? ` — ${run.skipped}` : ''}, so there is nothing to compare `
      + 'against eager. The search can still find what makes the compile itself throw.',
    broken: false,
  };
}

export function BisectPanel() {
  const running = useStore(s => s.bisecting);
  const result = useStore(s => s.bisect);
  const compile = useStore(s => s.result);
  const probes = useStore(s => s.bisectProbes);
  const note = useStore(s => s.bisectNote);
  const status = useStore(s => s.status);
  const [tolerance, setTolerance] = useState<number>(DEFAULT_TOLERANCE);

  const symptom = symptomOf(compile, tolerance);
  const ranPasses = compile ? new Set(compile.steps.filter(s => s.kind === 'pass').map(s => s.pass)).size : 0;

  return (
    <div className="bisect">
      <section className="memory-verdict">
        <h2>{result ? whatBroke(result) : symptom.headline}</h2>
        <p>{result ? result.conclusion : symptom.blurb}</p>
      </section>

      <div className="bisect-run">
        <label className="control">
          <span>Tolerance</span>
          <select
            value={tolerance}
            disabled={running}
            onChange={event => setTolerance(Number(event.target.value))}
          >
            {TOLERANCES.map(value => <option key={value} value={value}>{value.toExponential(0)}</option>)}
          </select>
        </label>
        <button
          className="run"
          disabled={running || status === 'compiling' || status === 'starting' || compile === null}
          title={symptom.broken
            ? 'turn passes off in a delta-debugging search until the smallest breaking set is named'
            : 'nothing is currently reproducing a failure — the search would confirm that the hard way'}
          onClick={() => { void actions.runBisect(tolerance); }}
        >
          {running ? 'Searching…' : symptom.broken ? 'Find the pass' : 'Search anyway'}
        </button>
        {running
          ? <span className="bisect-note" role="status">{note}</span>
          : ranPasses > 0 && (
            <span className="bisect-note">
              {plural(ranPasses, 'pass', 'es')} ran · about {Math.ceil(Math.log2(ranPasses + 1)) * 2} compiles, a few seconds each
            </span>
          )}
      </div>

      {result && result.culprits.length > 0 && (
        <section className="bisect-verdict">
          <ul>
            {result.culprits.map(name => (
              <li key={name}>
                <strong>{name}</strong>
                <em>{passLabel(name)}</em>
              </li>
            ))}
          </ul>
          <button className="pin" onClick={() => actions.turnOffCulprits()}>
            turn {result.culprits.length === 1 ? 'it' : 'them'} off and compile again
          </button>
        </section>
      )}

      {probes.length > 0 && (
        <table className="bisect-table">
          <thead>
            <tr>
              <th>probe</th>
              <th>passes off</th>
              <th>worst diff</th>
              <th>ms</th>
              <th>verdict</th>
            </tr>
          </thead>
          <tbody>
            {probes.map(probe => <Row key={probe.index} probe={probe} total={result ? result.candidates.length : 0} />)}
          </tbody>
        </table>
      )}

      {result && result.candidates.length > 0 && (
        <p className="bisect-footnote">
          {probes.length} compiles over {result.candidates.length} passes, {(result.totalMs / 1000).toFixed(1)}s in total.
          A pass is only a candidate if it actually ran, so anything already turned off stays off in every probe.
        </p>
      )}
    </div>
  );
}

function Row({ probe, total }: { probe: BisectProbe; total: number }) {
  const off = probe.disabled.length;

  return (
    <tr className={probe.good ? 'better' : 'worse'}>
      <td>{probe.index === 0 ? 'baseline' : `#${probe.index}`}</td>
      <td className="bisect-off" title={probe.disabled.join(', ')}>
        {off === 0 ? 'none' : `${off}${total > 0 ? ` of ${total}` : ''}`}
      </td>
      <td>{diffText(probe)}</td>
      <td>{probe.ms.toFixed(0)}</td>
      <td>{probe.good ? 'fixed' : 'still broken'}</td>
    </tr>
  );
}
