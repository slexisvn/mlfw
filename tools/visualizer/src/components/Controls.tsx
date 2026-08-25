import { useState } from 'react';
import { EXAMPLES } from '../examples/index.js';
import { RUN_SHORTCUT } from '../platform.js';
import { actions, isStale, useStore } from '../store.js';
import { passLabel } from '../catalog/naming.js';
import { TARGETS, targetNote } from '../catalog/targets.js';
import type { CompileOptions, TargetName } from '../protocol.js';

const STRATEGIES: { id: CompileOptions['fusionStrategy']; note: string }[] = [
  { id: 'priority', note: 'merge the most profitable pair first' },
  { id: 'dominator', note: 'merge along the graph’s dominator tree' },
  { id: 'greedy', note: 'merge the first legal pair found' },
];

const TOGGLES: { key: keyof CompileOptions; label: string; note: string }[] = [
  { key: 'fusion', label: 'fuse', note: 'merge neighbouring ops into one kernel' },
  { key: 'scheduling', label: 'schedule', note: 'choose loop order, tiling and threads' },
  { key: 'layout', label: 'layout', note: 'pick memory layouts and insert transposes' },
];

const COPY_RESET_MS = 1600;

export function Controls() {
  const options = useStore(s => s.options);
  const exampleId = useStore(s => s.exampleId);
  const status = useStore(s => s.status);
  const stale = useStore(isStale);
  const hasResult = useStore(s => s.result !== null);
  const [copied, setCopied] = useState(false);
  const example = EXAMPLES.find(e => e.id === exampleId);
  const edited = exampleId === '';
  const target = targetNote(options.target);
  const strategy = STRATEGIES.find(s => s.id === options.fusionStrategy) as (typeof STRATEGIES)[number];

  const copyLink = async (): Promise<void> => {
    await navigator.clipboard.writeText(actions.share());
    setCopied(true);
    setTimeout(() => setCopied(false), COPY_RESET_MS);
  };

  return (
    <div className="controls">
      <div className="control-row">
        <label className="control">
          <span>Example</span>
          <select value={exampleId} onChange={e => actions.loadExample(e.target.value)}>
            {edited && <option value="">— your own code —</option>}
            {EXAMPLES.map(e => <option key={e.id} value={e.id}>{e.title}</option>)}
          </select>
        </label>
        <button
          className={stale ? 'run stale' : 'run'}
          disabled={status === 'compiling' || status === 'starting'}
          onClick={() => { void actions.run(); }}
        >
          {status === 'compiling' ? 'Compiling…' : `Run ${RUN_SHORTCUT}`}
        </button>
      </div>

      <p className="blurb">
        {edited
          ? 'Your own code. End it with run(model, inputs) and press Run.'
          : (example as (typeof EXAMPLES)[number]).blurb}
      </p>

      {stale && (
        <p className="stale-note" role="status">
          You changed something since the last compile — press Run to update what is on screen.
        </p>
      )}

      <div className="control-row">
        <label className="control">
          <span>Target</span>
          <select
            value={options.target}
            onChange={e => actions.setOptions({ target: e.target.value as TargetName })}
          >
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
            {STRATEGIES.map(s => <option key={s.id} value={s.id}>{s.id}</option>)}
          </select>
        </label>
      </div>

      <p className="control-note">
        {target.note}
        {options.fusion ? ` · ${strategy.note}` : ' · fusion is off'}
      </p>

      <div className="toggles">
        {TOGGLES.map(toggle => (
          <Toggle
            key={toggle.key}
            label={toggle.label}
            note={toggle.note}
            on={options[toggle.key] as boolean}
            onChange={value => actions.setOptions({ [toggle.key]: value } as Partial<CompileOptions>)}
          />
        ))}
      </div>

      {options.disabledPasses.length > 0 && (
        <div className="disabled-passes">
          <span>turned off:</span>
          {options.disabledPasses.map(name => (
            <button
              key={name}
              title={`put ${passLabel(name)} back and recompile`}
              onClick={() => actions.togglePass(name)}
            >
              {name} ↺
            </button>
          ))}
        </div>
      )}

      {hasResult && (
        <button className="share" onClick={() => { void copyLink(); }}>
          {copied ? 'link copied' : 'copy a link to this'}
        </button>
      )}
    </div>
  );
}

function Toggle(
  { label, note, on, onChange }: { label: string; note: string; on: boolean; onChange: (v: boolean) => void },
) {
  return (
    <label className={on ? 'toggle on' : 'toggle'} title={note}>
      <input type="checkbox" checked={on} onChange={e => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}
