import { actions, useStore } from '../store.js';
import { ANOMALY_NOTES, anomaliesOf, profileOf } from '../catalog/pipeline.js';
import { agree, levelBadge, phaseLabel, plural } from '../catalog/naming.js';
import type { Anomaly, PassCost, PhaseCost } from '../catalog/pipeline.js';

const SLOW_SHARE = 0.1;

function share(ms: number, total: number): number {
  return total > 0 ? ms / total : 0;
}

export function ProfilePanel() {
  const result = useStore(s => s.result);

  if (!result) {
    return <div className="pane-empty">Run a compile and the time each pass took shows up here.</div>;
  }

  const profile = profileOf(result);
  const anomalies = anomaliesOf(result);
  const unaccounted = profile.totalMs - profile.measuredMs;

  return (
    <div className="profile">
      <section className="memory-verdict">
        <h2>where the {profile.totalMs.toFixed(0)}ms went</h2>
        <p>
          {profile.measuredMs.toFixed(0)}ms of it is inside passes; the other {unaccounted.toFixed(0)}ms is
          tracing, lowering, codegen and the instrumentation this view costs. A pass over{' '}
          {(SLOW_SHARE * 100).toFixed(0)}% of pass time is marked — that is where a compile-time regression
          lives, and the count beside it says how often the work was wasted.
        </p>
      </section>

      {anomalies.length > 0 && (
        <section className="result-block">
          <h3>{plural(anomalies.length, 'thing')} the pipeline did that it should not have<span>each one is a real bug or a real waste</span></h3>
          <div className="anomaly-list">
            {anomalies.map((anomaly, i) => <AnomalyRow key={i} anomaly={anomaly} />)}
          </div>
        </section>
      )}

      <section className="result-block">
        <h3>by phase<span>the pipeline&rsquo;s own stages</span></h3>
        <table className="bisect-table">
          <thead><tr><th>phase</th><th>level</th><th>ms</th><th>share</th><th>runs</th><th>changed</th></tr></thead>
          <tbody>
            {profile.phases.map(phase => <PhaseRow key={`${phase.phase}-${phase.level}`} phase={phase} total={profile.measuredMs} />)}
          </tbody>
        </table>
      </section>

      <section className="result-block">
        <h3>by pass<span>summed over every run of that pass</span></h3>
        <table className="bisect-table">
          <thead><tr><th>pass</th><th>ms</th><th>share</th><th>runs</th><th>changed</th></tr></thead>
          <tbody>
            {profile.passes.map(pass => <PassRow key={`${pass.pass}-${pass.level}`} pass={pass} total={profile.measuredMs} />)}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function Bar({ fraction }: { fraction: number }) {
  return (
    <span className="cost-bar" aria-hidden="true">
      <span style={{ width: `${Math.min(100, fraction * 100).toFixed(1)}%` }} />
    </span>
  );
}

function PhaseRow({ phase, total }: { phase: PhaseCost; total: number }) {
  const fraction = share(phase.ms, total);
  return (
    <tr className={fraction >= SLOW_SHARE ? 'worse' : ''}>
      <td>{phaseLabel(phase.phase)}</td>
      <td>{levelBadge(phase.level)}</td>
      <td>{phase.ms.toFixed(1)}</td>
      <td><Bar fraction={fraction} /></td>
      <td>{phase.runs}</td>
      <td>{phase.changed}</td>
    </tr>
  );
}

function PassRow({ pass, total }: { pass: PassCost; total: number }) {
  const fraction = share(pass.ms, total);
  return (
    <tr className={fraction >= SLOW_SHARE ? 'worse' : ''}>
      <td>{pass.pass}</td>
      <td>{pass.ms.toFixed(1)}</td>
      <td><Bar fraction={fraction} /></td>
      <td>{pass.runs}</td>
      <td className={pass.changed === 0 ? 'never-fired' : ''}>{pass.changed}</td>
    </tr>
  );
}

function AnomalyRow({ anomaly }: { anomaly: Anomaly }) {
  const note = ANOMALY_NOTES[anomaly.kind];
  return (
    <button className={`anomaly ${anomaly.kind}`} onClick={() => actions.select(anomaly.step)}>
      <span className="anomaly-label">{note.label}</span>
      <span className="anomaly-pass">{anomaly.pass}</span>
      <span className="anomaly-detail">{anomaly.detail}</span>
      <span className="anomaly-why">{note.why}</span>
    </button>
  );
}
