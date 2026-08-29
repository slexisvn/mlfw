import { actions, interpretableSteps, useStore } from '../store.js';
import { agree, levelLabel, phaseLabel, plural } from '../catalog/naming.js';
import type { CellDiff, CompileStep, SemanticReport } from '../protocol.js';

function num(value: number): string {
  if (Number.isInteger(value)) return String(value);
  const size = Math.abs(value);
  return size >= 1e4 || size < 1e-3 ? value.toExponential(3) : value.toFixed(6);
}

function toneOf(report: SemanticReport): string {
  if (!report.ran || report.truncated || report.storageReused) return 'unknown';
  if (report.changedCount > 0) return 'broken';
  if (report.droppedCount > 0 || report.addedCount > 0) return 'suspect';
  return 'clean';
}

export function SemanticsPanel({ step }: { step: CompileStep }) {
  const stored = useStore(s => s.semantics[step.index]);
  const pending = useStore(s => s.semanticsPending);
  const running = pending === step.index;
  const report = stored ? stored.report : null;

  return (
    <div className="semantics">
      <section className="memory-verdict">
        <h2>does this pass still compute the same thing?</h2>
        <p>
          The loop nest is run before and after the pass over its whole iteration space, with every load
          modelled as a deterministic value of the cell it reads. Comparing the two store maps decides it:
          a pass that only moves loops writes the same cells with the same values, and a pass that changed
          the arithmetic or an index does not. Nothing is compiled and no kernel runs, so it answers in
          milliseconds where a GPU round trip takes minutes.
        </p>
      </section>

      {!step.interpretable ? (
        <NoNest reason={stored?.unavailable ?? null} />
      ) : !stored ? (
        <div className="bisect-run">
          <button className="run" disabled={running} onClick={() => { void actions.proveStep(step.index); }}>
            {running ? 'Interpreting…' : `Interpret ${step.pass}`}
          </button>
          {running && <span className="bisect-note" role="status">running both sides over the full iteration space</span>}
        </div>
      ) : report === null ? (
        <p className="pane-empty">{stored.unavailable}</p>
      ) : (
        <>
          <section className={`bisect-verdict semantic ${toneOf(report)}`}>
            <p>{report.verdict}</p>
          </section>

          <dl className="extras semantic-counts">
            <div><dt>cells compared</dt><dd>{report.compared}</dd></div>
            <div><dt>stores before</dt><dd>{report.storesBefore}</dd></div>
            <div><dt>stores after</dt><dd>{report.storesAfter}</dd></div>
            <div><dt>write order</dt><dd>{report.reordered ? 'changed' : 'identical'}</dd></div>
            <div><dt>interpreted in</dt><dd>{stored.ms.toFixed(0)}ms</dd></div>
          </dl>

          {report.storageReused && (
            <Names
              title={`${plural(report.vanishedBuffers.length, 'buffer')} folded into another's storage`}
              names={report.vanishedBuffers}
            />
          )}
          {report.changedCount > 0 && (
            <Cells title={`${plural(report.changedCount, 'cell')} hold a different value`} rows={report.changed} />
          )}
          {report.droppedCount > 0 && (
            <Names title={`${plural(report.droppedCount, 'cell')} ${agree(report.droppedCount, 'is')} no longer written`} names={report.dropped} />
          )}
          {report.addedCount > 0 && (
            <Names title={`${plural(report.addedCount, 'cell')} ${agree(report.addedCount, 'is')} newly written`} names={report.added} />
          )}
          {report.newBuffers.length > 0 && (
            <Names title={`${plural(report.newBuffers.length, 'buffer')} this pass introduced`} names={report.newBuffers} />
          )}
        </>
      )}
    </div>
  );
}

function NoNest({ reason }: { reason: string | null }) {
  const candidates = useStore(interpretableSteps);

  if (candidates.length === 0) {
    return (
      <p className="pane-empty">
        {reason ?? 'Nothing in this compile reached a loop nest, so there is no schedule to prove yet. '
          + 'Passes over the tensor IR and the low-level IR are the ones this can check.'}
      </p>
    );
  }

  return (
    <section className="result-block">
      <h3>
        this step has no loop nest
        <span>{reason ?? 'only passes that rewrite a loop nest can be interpreted — these are the ones in this compile'}</span>
      </h3>
      <div className="jump-list">
        {candidates.map(candidate => (
          <button key={candidate.index} className="jump" onClick={() => actions.select(candidate.index)}>
            <span className="jump-pass">{candidate.pass}</span>
            <span className="jump-phase">{phaseLabel(candidate.phase)} · {levelLabel(candidate.level)}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function Cells({ title, rows }: { title: string; rows: readonly CellDiff[] }) {
  return (
    <section className="result-block">
      <h3>{title}<span>the first few, in write order</span></h3>
      <table className="bisect-table">
        <thead><tr><th>cell</th><th>before</th><th>after</th></tr></thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.cell} className="worse">
              <td>{row.cell}</td>
              <td>{num(row.before)}</td>
              <td>{num(row.after)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Names({ title, names }: { title: string; names: readonly string[] }) {
  return (
    <section className="result-block">
      <h3>{title}<span>the first few</span></h3>
      <ul className="invariant-list">
        {names.map(name => <li key={name}>{name}</li>)}
      </ul>
    </section>
  );
}
