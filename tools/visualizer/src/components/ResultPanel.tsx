import { useStore } from '../store.js';
import { targetNote } from '../catalog/targets.js';
import type { RunResult, TensorPreview } from '../protocol.js';

const CLOSE_ENOUGH = 1e-4;

export function ResultPanel({ run }: { run: RunResult | null }) {
  const target = useStore(s => (s.ranOptions ? s.ranOptions.target : null));
  const ranOn = target ? targetNote(target).label : null;

  if (!run || (!run.ran && !run.skipped && !run.error)) {
    return <div className="pane-empty">Run a compile and the kernel it produced gets executed here.</div>;
  }

  if (run.skipped) {
    return (
      <div className="result">
        <Verdict tone="skipped" headline="not executed here" note={run.skipped} ranOn={ranOn} />
      </div>
    );
  }

  if (run.error) {
    return (
      <div className="result">
        <Verdict tone="off" headline="the kernel failed to run" note={run.error} ranOn={ranOn} />
      </div>
    );
  }

  const exact = run.maxAbsDiff === 0;
  const close = run.maxAbsDiff !== null && run.maxAbsDiff < CLOSE_ENOUGH;
  const speedup = run.eagerMs && run.compiledMs ? run.eagerMs / run.compiledMs : null;
  const faster = speedup !== null && speedup >= 1;

  return (
    <div className="result">
      <Verdict
        tone={run.maxAbsDiff === null ? 'skipped' : exact ? 'exact' : close ? 'close' : 'off'}
        ranOn={ranOn}
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
        <h3>
          timing
          <span>
            per call, averaged over {run.iterations} call{run.iterations === 1 ? '' : 's'}
          </span>
        </h3>
        <div className="timings">
          <Timing label="compiled" ms={run.compiledMs} highlight={faster} />
          <Timing label="eager" ms={run.eagerMs} highlight={speedup !== null && !faster} />
          {speedup !== null && (
            <div className={faster ? 'speedup faster' : 'speedup slower'}>
              {faster ? `${speedup.toFixed(2)}× faster` : `${(1 / speedup).toFixed(2)}× slower`}
            </div>
          )}
        </div>
        {speedup !== null && !faster && (
          <p className="timing-note">
            The compiled path loses on a model this small: launching a kernel costs a fixed amount that a
            few hundred numbers cannot pay back. Try a bigger shape and watch the gap flip.
          </p>
        )}
      </section>

      {run.inputs.length > 0 && (
        <section className="result-block">
          <h3>inputs<span>what the model was fed</span></h3>
          {run.inputs.map((tensor, i) => <TensorRow key={i} tensor={tensor} />)}
        </section>
      )}

      <section className="result-block">
        <h3>output<span>the compiled kernel, then the same model run op by op</span></h3>
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

function Verdict(
  { tone, headline, note, ranOn }: { tone: string; headline: string; note: string; ranOn: string | null },
) {
  return (
    <section className={`verdict-card ${tone}`}>
      <h2>{headline}</h2>
      <p>{note}</p>
      {ranOn && <p className="ran-on">target: {ranOn}</p>}
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
