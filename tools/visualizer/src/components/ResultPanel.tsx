import type { RunResult, TensorPreview } from '../protocol.js';

export function ResultPanel({ run }: { run: RunResult | null }) {
  if (!run || (!run.ran && !run.skipped && !run.error)) {
    return <div className="pane-empty">Run a compile to execute the kernel it produced.</div>;
  }

  if (run.skipped) {
    return (
      <div className="result">
        <Verdict tone="skipped" headline="not executed here" note={run.skipped} />
      </div>
    );
  }

  if (run.error) {
    return (
      <div className="result">
        <Verdict tone="off" headline="the kernel failed to run" note={run.error} />
      </div>
    );
  }

  const exact = run.maxAbsDiff === 0;
  const close = run.maxAbsDiff !== null && run.maxAbsDiff < 1e-4;
  const speedup = run.eagerMs && run.compiledMs ? run.eagerMs / run.compiledMs : null;

  return (
    <div className="result">
      <Verdict
        tone={run.maxAbsDiff === null ? 'skipped' : exact ? 'exact' : close ? 'close' : 'off'}
        headline={
          run.maxAbsDiff === null
            ? 'ran, but the outputs could not be compared'
            : exact
              ? 'bit-exact against eager'
              : `matches eager to ${run.maxAbsDiff.toExponential(1)}`
        }
        note={
          run.maxAbsDiff === null || exact
            ? 'The compiled kernel and the same model run op by op produced the same values.'
            : close
              ? 'Within float32 rounding — the passes reordered arithmetic, they did not change the answer.'
              : 'Larger than float32 rounding explains. An optimization changed the result.'
        }
      />

      <section className="result-block">
        <h3>timing<span>per call, {run.iterations} iteration{run.iterations === 1 ? '' : 's'} per sample</span></h3>
        <div className="timings">
          <Timing label="compiled" ms={run.compiledMs} highlight={speedup !== null && speedup > 1} />
          <Timing label="eager" ms={run.eagerMs} highlight={speedup !== null && speedup < 1} />
          {speedup !== null && (
            <div className="speedup">
              {speedup >= 1 ? `${speedup.toFixed(2)}× faster` : `${(1 / speedup).toFixed(2)}× slower`}
            </div>
          )}
        </div>
      </section>

      {run.inputs.length > 0 && (
        <section className="result-block">
          <h3>inputs</h3>
          {run.inputs.map((tensor, i) => <TensorRow key={i} tensor={tensor} />)}
        </section>
      )}

      <section className="result-block">
        <h3>output<span>compiled, then eager</span></h3>
        {run.outputs.map((tensor, i) => (
          <div className="output-pair" key={i}>
            <TensorRow tensor={tensor} tag="compiled" />
            {run.eagerOutputs[i] && <TensorRow tensor={run.eagerOutputs[i]} tag="eager" muted />}
          </div>
        ))}
      </section>
    </div>
  );
}

function Verdict({ tone, headline, note }: { tone: string; headline: string; note: string }) {
  return (
    <section className={`verdict-card ${tone}`}>
      <h2>{headline}</h2>
      <p>{note}</p>
    </section>
  );
}

function Timing({ label, ms, highlight }: { label: string; ms: number | null; highlight: boolean }) {
  return (
    <div className={highlight ? 'timing-cell best' : 'timing-cell'}>
      <span className="timing-label">{label}</span>
      <span className="timing-value">{ms === null ? '—' : `${ms.toFixed(3)}ms`}</span>
    </div>
  );
}

function TensorRow({ tensor, tag, muted }: { tensor: TensorPreview; tag?: string; muted?: boolean }) {
  return (
    <div className={muted ? 'tensor-row muted' : 'tensor-row'}>
      {tag && <span className="tensor-tag">{tag}</span>}
      <span className="tensor-shape">[{tensor.shape.join(', ')}] {tensor.dtype}</span>
      <span className="tensor-values">
        {tensor.preview.map(v => Number(v.toFixed(4))).join(', ')}
        {tensor.numel > tensor.preview.length && ` … ${tensor.numel} values`}
      </span>
    </div>
  );
}
