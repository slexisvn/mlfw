import { EXAMPLES } from '../examples/index.js';
import { actions, useStore } from '../store.js';
import type { CompileOptions, TargetName } from '../protocol.js';

const TARGETS: { id: TargetName; label: string }[] = [
  { id: 'cpu', label: 'CPU (JS)' },
  { id: 'wasm', label: 'WebAssembly' },
  { id: 'cuda', label: 'CUDA' },
  { id: 'webgpu', label: 'WebGPU' },
];

const STRATEGIES: CompileOptions['fusionStrategy'][] = ['priority', 'dominator', 'greedy'];

export function Controls() {
  const options = useStore(s => s.options);
  const exampleId = useStore(s => s.exampleId);
  const status = useStore(s => s.status);
  const example = EXAMPLES.find(e => e.id === exampleId);

  return (
    <div className="controls">
      <div className="control-row">
        <label className="control">
          <span>Example</span>
          <select value={exampleId} onChange={e => actions.loadExample(e.target.value)}>
            {exampleId === '' && <option value="">(edited)</option>}
            {EXAMPLES.map(e => <option key={e.id} value={e.id}>{e.title}</option>)}
          </select>
        </label>
        <button
          className="run"
          disabled={status === 'compiling' || status === 'starting'}
          onClick={() => { void actions.run(); }}
        >
          {status === 'compiling' ? 'Compiling…' : 'Run ⌘⏎'}
        </button>
      </div>

      {example && <p className="blurb">{example.blurb}</p>}

      <div className="control-row">
        <label className="control">
          <span>Target</span>
          <select value={options.target} onChange={e => actions.setOptions({ target: e.target.value as TargetName })}>
            {TARGETS.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
        </label>
        <label className="control">
          <span>Fusion</span>
          <select
            value={options.fusionStrategy}
            disabled={!options.fusion}
            onChange={e => actions.setOptions({ fusionStrategy: e.target.value as CompileOptions['fusionStrategy'] })}
          >
            {STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
      </div>

      <div className="toggles">
        <Toggle label="fuse" on={options.fusion} onChange={v => actions.setOptions({ fusion: v })} />
        <Toggle label="schedule" on={options.scheduling} onChange={v => actions.setOptions({ scheduling: v })} />
        <Toggle label="layout" on={options.layout} onChange={v => actions.setOptions({ layout: v })} />
      </div>

      {options.disabledPasses.length > 0 && (
        <div className="disabled-passes">
          <span>off:</span>
          {options.disabledPasses.map(name => (
            <button key={name} onClick={() => actions.togglePass(name)}>{name} ✕</button>
          ))}
        </div>
      )}
    </div>
  );
}

function Toggle({ label, on, onChange }: { label: string; on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className={on ? 'toggle on' : 'toggle'}>
      <input type="checkbox" checked={on} onChange={e => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}
