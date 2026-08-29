import { actions, useStore } from '../store.js';
import { METRICS, formatMetric, optionDiffs } from '../catalog/metrics.js';
import { diffLedgers, ledgerOf, parseLedger, summarize } from '../catalog/ledger.js';
import { download } from '../repro.js';
import type { Ledger, LedgerEntry } from '../catalog/ledger.js';
import type { Metric } from '../catalog/metrics.js';
import type { CompileResponse } from '../protocol.js';

const RATIO_FLOOR = 1e-9;
const NOTICEABLE = 0.02;

type Direction = 'better' | 'worse' | 'same' | 'unknown';

function directionOf(metric: Metric, from: number | null, to: number | null): Direction {
  if (from === null || to === null) return 'unknown';
  if (from === to) return 'same';
  const improved = metric.lowerIsBetter ? to < from : to > from;
  return improved ? 'better' : 'worse';
}

function deltaOf(metric: Metric, from: number | null, to: number | null): string {
  if (from === null || to === null || from === to) return '';
  if (metric.kind === 'diff') return to > from ? 'further off' : 'closer';
  if (Math.abs(from) < RATIO_FLOOR) return 'from nothing';
  const ratio = to / from;
  if (Math.abs(ratio - 1) < NOTICEABLE) return 'about the same';
  return ratio > 1 ? `${ratio.toFixed(2)}× more` : `${(1 / ratio).toFixed(2)}× less`;
}

export function ComparePanel() {
  const result = useStore(s => s.result);
  const baseline = useStore(s => s.baseline);
  const options = useStore(s => s.options);
  const source = useStore(s => s.source);

  if (!result) {
    return <div className="pane-empty">Run a compile first, then pin it as the run everything else is measured against.</div>;
  }

  if (!baseline) {
    return (
      <div className="compare empty">
        <section className="memory-verdict">
          <h2>nothing pinned yet</h2>
          <p>
            Turning a pass off tells you what changed in the IR. It does not tell you what that cost.
            Pin this compile, change one thing, and both runs stay on screen side by side.
          </p>
        </section>

        <section className="baseline">
          <h3>
            the pinned run
            <span>measures this run against another one from this session — time, op counts, accuracy. Kept in memory, gone when you reload.</span>
          </h3>
          <div className="compare-actions">
            <button className="pin" onClick={() => actions.pinBaseline()}>pin this run as the baseline</button>
          </div>
        </section>

        <LedgerSection />
      </div>
    );
  }

  const diffs = optionDiffs(baseline.options, options);
  const sameSource = baseline.source === source;

  return (
    <div className="compare">
      <section className="memory-verdict">
        <h2>{diffs.length === 0 && sameSource ? 'the same settings, run twice' : 'one thing changed'}</h2>
        <p>
          {sameSource
            ? 'Both runs compiled the same source.'
            : 'The source changed between the two runs, so the numbers below are not a clean counterfactual.'}
          {diffs.length === 0
            ? ' The settings are identical too — any difference in the numbers is measurement noise, which is worth knowing before you trust a small one.'
            : ''}
        </p>
        {diffs.length > 0 && (
          <ul className="compare-diffs">
            {diffs.map(diff => (
              <li key={diff.key}>
                <span className="compare-key">{diff.key}</span>
                <span className="compare-from">{diff.from}</span>
                <span className="compare-arrow">→</span>
                <span className="compare-to">{diff.to}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <table className="compare-table">
        <thead>
          <tr>
            <th>measure</th>
            <th>pinned</th>
            <th>now</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {METRICS.map(metric => (
            <Row key={metric.id} metric={metric} from={baseline.response} to={result} />
          ))}
        </tbody>
      </table>

      <div className="compare-actions">
        <button className="pin" onClick={() => actions.pinBaseline()}>pin the current run instead</button>
        <button className="unpin" onClick={() => actions.clearBaseline()}>forget the pinned run</button>
      </div>

      <hr className="compare-split" />

      <LedgerSection />
    </div>
  );
}

function LedgerSection() {
  const result = useStore(s => s.result);
  const source = useStore(s => s.source);
  const options = useStore(s => s.options);
  const saved = useStore(s => s.ledger);

  if (!result) return null;
  const now = ledgerOf(result, source, options);
  const diff = saved ? diffLedgers(saved, now) : null;

  const load = (file: File | undefined): void => {
    if (!file) return;
    void file.text().then(text => actions.setLedger(parseLedger(text)));
  };

  return (
    <section className="ledger">
      <h3>
        the pass ledger
        <span>
          records what every pass <em>did</em> — its op counts and outcome — as a file you can keep. Diff a ledger
          from another machine or an older build to find the pass whose behaviour moved.
        </span>
      </h3>

      <div className="compare-actions">
        <button className="pin" onClick={() => download('pass-ledger.json', JSON.stringify(now, null, 2))}>
          save this ledger ({now.entries.length} runs)
        </button>
        <label className="unpin file">
          load a ledger to diff
          <input type="file" accept="application/json" onChange={event => load(event.target.files?.[0])} />
        </label>
        {saved && <button className="unpin" onClick={() => actions.setLedger(null)}>forget it</button>}
      </div>

      {saved && diff && <LedgerDiffView saved={saved} diff={diff} />}
    </section>
  );
}

function LedgerDiffView({ saved, diff }: { saved: Ledger; diff: ReturnType<typeof diffLedgers> }) {
  const clean = diff.changed.length === 0 && diff.onlyOld.length === 0 && diff.onlyNew.length === 0;

  return (
    <>
      <p className={clean ? 'ledger-verdict clean' : 'ledger-verdict dirty'}>
        {summarize(diff)} <em>saved {new Date(saved.savedAt).toLocaleString()}</em>
      </p>

      {diff.changed.length > 0 && (
        <table className="bisect-table">
          <thead><tr><th>pass run</th><th>was</th><th>now</th><th>outcome</th></tr></thead>
          <tbody>
            {diff.changed.map(change => (
              <tr key={change.key} className="worse">
                <td>{change.entry.pass}{change.entry.run > 1 && ` #${change.entry.run}`}</td>
                <td>{change.was.before}→{change.was.after}</td>
                <td>{change.entry.before}→{change.entry.after}</td>
                <td>{change.was.outcome === change.entry.outcome ? change.entry.outcome : `${change.was.outcome} → ${change.entry.outcome}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {diff.onlyOld.length > 0 && <Missing title="runs that no longer happen" entries={diff.onlyOld} />}
      {diff.onlyNew.length > 0 && <Missing title="runs that are new" entries={diff.onlyNew} />}
    </>
  );
}

function Missing({ title, entries }: { title: string; entries: readonly LedgerEntry[] }) {
  return (
    <div className="ledger-missing">
      <h4>{title}</h4>
      <ul className="invariant-list">
        {entries.map(entry => (
          <li key={entry.key}>{entry.pass} #{entry.run} · {entry.phase} · {entry.before}→{entry.after}</li>
        ))}
      </ul>
    </div>
  );
}

function Row({ metric, from, to }: { metric: Metric; from: CompileResponse; to: CompileResponse }) {
  const before = metric.of(from);
  const after = metric.of(to);
  const direction = directionOf(metric, before, after);

  return (
    <tr className={direction}>
      <th scope="row" title={metric.meaning}>{metric.label}</th>
      <td>{formatMetric(metric.kind, before)}</td>
      <td>{formatMetric(metric.kind, after)}</td>
      <td className="compare-delta">{deltaOf(metric, before, after)}</td>
    </tr>
  );
}
