import { useStore } from '../store.js';
import { targetNote } from '../catalog/targets.js';
import { graphCostOf } from '../catalog/metrics.js';
import type { RunResult, TensorPreview } from '../protocol.js';

const CLOSE_ENOUGH = 1e-4;

function countBad(groups: readonly (readonly TensorPreview[])[]): { nan: number; inf: number } {
  let nan = 0;
  let inf = 0;
  for (const group of groups) {
    for (const tensor of group) {
      nan += tensor.stats.nan;
      inf += tensor.stats.inf;
    }
  }
  return { nan, inf };
}

function listBad(bad: { nan: number; inf: number }): string {
  return [bad.nan > 0 ? `${bad.nan} NaN` : '', bad.inf > 0 ? `${bad.inf} Inf` : '']
    .filter(Boolean).join(' and ');
}

export function ResultPanel({ run }: { run: RunResult | null }) {
  const target = useStore(s => (s.ranOptions ? s.ranOptions.target : null));
  const result = useStore(s => s.result);
  const ranOn = target ? targetNote(target).label : null;
  const cost = result ? graphCostOf(result) : null;

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

  const trained = run.gradients.length > 0;
  const worst = trained && run.maxAbsGradDiff !== null && run.maxAbsDiff !== null
    ? Math.max(run.maxAbsDiff, run.maxAbsGradDiff)
    : run.maxAbsDiff;
  const exact = worst === 0;
  const close = worst !== null && worst < CLOSE_ENOUGH;
  const speedup = run.eagerMs && run.compiledMs ? run.eagerMs / run.compiledMs : null;
  const faster = speedup !== null && speedup >= 1;

  const bad = countBad([run.outputs, run.gradients]);
  const eagerBad = countBad([run.eagerOutputs, run.eagerGradients]);
  const sick = bad.nan + bad.inf > 0;

  return (
    <div className="result">
      <Verdict
        tone={sick ? 'off' : worst === null ? 'skipped' : exact ? 'exact' : close ? 'close' : 'off'}
        ranOn={ranOn}
        headline={
          sick
            ? `${listBad(bad)} came out of the ${trained ? 'training step' : 'model'}`
            : worst === null
              ? `ran, but the ${trained ? 'gradients' : 'outputs'} could not be compared`
              : exact
                ? `bit-exact against eager${trained ? ', gradients included' : ''}`
                : `matches eager to ${worst.toExponential(1)}`
        }
        note={
          sick
            ? eagerBad.nan + eagerBad.inf > 0
              ? 'Eager autograd produced them too, so the model itself is doing this — an optimization is not to blame. The Health tab says which layer they start at.'
              : 'Eager autograd stayed finite on the same inputs, so a pass introduced them. Turn passes off one at a time to find which.'
            : worst === null || exact
              ? trained
                ? 'The compiled training step and the same step run op by op through autograd produced the same values and the same gradients.'
                : 'The compiled kernel and the same model run op by op produced the same values.'
              : close
                ? 'Within float32 rounding — the passes reordered arithmetic, they did not change the answer.'
                : 'Larger than float32 rounding explains. An optimization changed the result.'
        }
      />

      <section className="result-block">
        <h3>
          timing
          <span>
            per {trained ? 'training step' : 'call'}, averaged over {run.iterations}{' '}
            {trained ? 'step' : 'call'}{run.iterations === 1 ? '' : 's'}
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

      {cost && run.compiledMs !== null && <Roofline cost={cost} ms={run.compiledMs} />}

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

      {trained && (
        <section className="result-block">
          <h3>
            gradients
            <span>one per input, then one per parameter — compiled, then eager autograd</span>
          </h3>
          {run.gradients.map((tensor, i) => (
            <div className="output-pair" key={i}>
              <TensorRow tensor={tensor} tag="compiled" />
              {run.eagerGradients[i] && <TensorRow tensor={run.eagerGradients[i]} tag="eager" muted />}
            </div>
          ))}
          {run.maxAbsGradDiff !== null && (
            <p className="timing-note">
              Worst gradient disagreement: {run.maxAbsGradDiff.toExponential(1)}. The backward graph was
              derived from the forward one by rule, so this number is the only evidence that the derivation
              and the optimizations applied to it are both right.
            </p>
          )}
        </section>
      )}
    </div>
  );
}

const MS_PER_SECOND = 1000;
const GIGA = 1e9;

function Roofline({ cost, ms }: { cost: { bytes: number; flops: number }; ms: number }) {
  const seconds = ms / MS_PER_SECOND;
  const bandwidth = cost.bytes / seconds / GIGA;
  const throughput = cost.flops / seconds / GIGA;
  const intensity = cost.flops / cost.bytes;
  const machineRatio = throughput / bandwidth;

  return (
    <section className="result-block">
      <h3>where the time went<span>the two coordinates a roofline is drawn in</span></h3>
      <div className="timings">
        <Timing label="moved" ms={null} value={`${bandwidth.toFixed(2)} GB/s`} />
        <Timing label="computed" ms={null} value={`${throughput.toFixed(2)} GFLOP/s`} />
        <Timing label="intensity" ms={null} value={`${intensity.toFixed(2)} flop/byte`} />
      </div>
      <p className="timing-note">
        The graph reads and writes {(cost.bytes / 1024).toFixed(1)} KB and does{' '}
        {(cost.flops / 1000).toFixed(1)}k floating-point operations, so it asks the machine for{' '}
        {machineRatio.toFixed(2)} flops per byte. A machine that can do more than that per byte finishes
        this program waiting on memory, and no amount of faster arithmetic will help it — that is the
        argument fusion is making when it cuts the traffic.
      </p>
    </section>
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

function Timing(
  { label, ms, highlight, value }: { label: string; ms: number | null; highlight?: boolean; value?: string },
) {
  return (
    <div className={highlight ? 'timing-cell best' : 'timing-cell'}>
      <span className="timing-label">{label}</span>
      <span className="timing-value">{value ?? (ms === null ? '—' : `${ms.toFixed(3)}ms`)}</span>
    </div>
  );
}

function TensorRow({ tensor, tag, muted }: { tensor: TensorPreview; tag?: string; muted?: boolean }) {
  const { stats } = tensor;
  return (
    <div className={muted ? 'tensor-row muted' : 'tensor-row'}>
      {tag && <span className="tensor-tag">{tag}</span>}
      <span className="tensor-shape">[{tensor.shape.join(', ')}] {tensor.dtype}</span>
      <span className="tensor-values">
        {tensor.preview.map(v => Number(v.toFixed(4))).join(', ')}
        {tensor.numel > tensor.preview.length && ` … ${tensor.numel} values`}
      </span>
      {stats.nan > 0 && <span className="flag bad">{stats.nan} NaN</span>}
      {stats.inf > 0 && <span className="flag bad">{stats.inf} Inf</span>}
    </div>
  );
}
