import { actions, useStore } from '../store.js';
import { METRICS, formatMetric, optionDiffs } from '../catalog/metrics.js';
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
        <button className="pin" onClick={() => actions.pinBaseline()}>pin this run as the baseline</button>
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
