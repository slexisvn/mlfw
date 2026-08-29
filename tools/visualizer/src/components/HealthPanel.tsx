import { actions } from '../store.js';
import type { LayerActivation, RunResult, TensorPreview, TensorStats } from '../protocol.js';

const EXPLODED = 1e4;
const VANISHED = 1e-7;

function num(value: number | null): string {
  if (value === null) return '—';
  if (value === 0) return '0';
  const size = Math.abs(value);
  return size >= 1e4 || size < 1e-3 ? value.toExponential(1) : value.toFixed(3);
}

function Flags({ stats }: { stats: TensorStats }) {
  if (stats.nan === 0 && stats.inf === 0) return null;
  return (
    <>
      {stats.nan > 0 && <span className="flag bad">{stats.nan} NaN</span>}
      {stats.inf > 0 && <span className="flag bad">{stats.inf} Inf</span>}
    </>
  );
}

function Shape({ tensor }: { tensor: TensorPreview }) {
  return <span className="health-shape">[{tensor.shape.join(', ')}] {tensor.dtype}</span>;
}

function Stats({ stats }: { stats: TensorStats }) {
  return (
    <span className="health-stats">
      <em>min</em> {num(stats.min)} <em>max</em> {num(stats.max)} <em>mean</em> {num(stats.mean)} <em>sd</em> {num(stats.std)}
    </span>
  );
}

function Activations({ layers }: { layers: readonly LayerActivation[] }) {
  return (
    <section className="result-block">
      <h3>activations<span>every leaf layer, in the order the forward pass ran them</span></h3>
      <div className="health-table">
        {layers.map((layer, i) => layer.outputs.map((tensor, j) => (
          <div
            key={`${i}-${j}`}
            className={layer.line === null ? 'health-row' : 'health-row traceable'}
            onPointerEnter={event => { if (event.pointerType === 'mouse') actions.focusSource(layer.line); }}
            onPointerLeave={event => { if (event.pointerType === 'mouse') actions.focusSource(null); }}
            onClick={() => actions.focusSource(layer.line)}
          >
            <span className="health-name">
              {layer.name}{tensor.name && ` · ${tensor.name}`}
              <em>{layer.kind}</em>
            </span>
            <Shape tensor={tensor} />
            <Stats stats={tensor.stats} />
            <span className="health-flags">
              <Flags stats={tensor.stats} />
              {tensor.numel > 0 && tensor.stats.zeros === tensor.numel && <span className="flag warn">all zero</span>}
            </span>
          </div>
        )))}
      </div>
    </section>
  );
}

function Gradients({ run }: { run: RunResult }) {
  const inputCount = run.inputs.length;

  return (
    <section className="result-block">
      <h3>gradients<span>one per input, then one per parameter — norm, and how big it is against the weight</span></h3>
      <div className="health-table">
        {run.gradients.map((grad, i) => {
          const param = i >= inputCount ? run.parameters[i - inputCount] : undefined;
          const ratio = param && param.stats.norm > 0 ? grad.stats.norm / param.stats.norm : null;
          return (
            <div className="health-row" key={grad.name}>
              <span className="health-name">{grad.name}</span>
              <Shape tensor={grad} />
              <span className="health-stats">
                <em>‖g‖</em> {num(grad.stats.norm)}
                {param && <> <em>‖w‖</em> {num(param.stats.norm)}</>}
                {ratio !== null && <> <em>ratio</em> {num(ratio)}</>}
              </span>
              <span className="health-flags">
                <Flags stats={grad.stats} />
                {ratio !== null && ratio > EXPLODED && <span className="flag bad">exploding</span>}
                {ratio !== null && ratio < VANISHED && <span className="flag warn">vanishing</span>}
                {grad.numel > 0 && grad.stats.zeros === grad.numel && <span className="flag warn">no gradient</span>}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function HealthPanel({ run }: { run: RunResult | null }) {
  if (!run || !run.ran) {
    return <div className="pane-empty">Run a compile and every layer reports the numbers it produced here.</div>;
  }

  if (run.layers.length === 0 && run.gradients.length === 0) {
    return (
      <div className="pane-empty">
        Nothing to inspect: this model is a plain function, so there are no layers to hook, and inference
        mode has no gradients. Build it out of nn modules, or switch Direction to a train mode.
      </div>
    );
  }

  return (
    <div className="result">
      {run.layers.length > 0 && <Activations layers={run.layers} />}
      {run.gradients.length > 0 && <Gradients run={run} />}
    </div>
  );
}
