import { useState } from 'react';
import { EXAMPLES } from '../examples/index.js';
import { RUN_SHORTCUT } from '../platform.js';
import { actions, isStale, useStore } from '../store.js';
import { passLabel, plural } from '../catalog/naming.js';
import { TARGETS, targetNote } from '../catalog/targets.js';
import { VERIFY_LEVELS, verifyNote } from '../catalog/diagnostics.js';
import { download, reproScript } from '../repro.js';
import type { BackwardMode, CompileOptions, TargetName, VerifyLevelName } from '../protocol.js';

const STRATEGIES: { id: CompileOptions['fusionStrategy']; note: string }[] = [
  { id: 'priority', note: 'merge the most profitable pair first' },
  { id: 'dominator', note: 'merge along the graph’s dominator tree' },
  { id: 'greedy', note: 'merge the first legal pair found' },
];

const TOGGLES: { key: keyof CompileOptions; label: string; note: string }[] = [
  { key: 'fusion', label: 'fuse', note: 'merge neighbouring ops into one kernel' },
  { key: 'scheduling', label: 'schedule', note: 'choose loop order, tiling and threads' },
  { key: 'autotune', label: 'tune', note: 'search for a schedule instead of deriving one from rules' },
  { key: 'layout', label: 'layout', note: 'pick memory layouts and insert transposes' },
];

const BACKWARD_MODES: { id: BackwardMode; label: string; note: string }[] = [
  { id: 'off', label: 'inference', note: 'compile the forward pass only' },
  {
    id: 'separate',
    label: 'train · separate',
    note: 'differentiate the graph and compile the backward as its own function, the way a training step usually runs',
  },
  {
    id: 'joint',
    label: 'train · joint',
    note: 'put forward and backward in one graph, so fusion can work across the boundary between them',
  },
];

export function Controls() {
  const options = useStore(s => s.options);
  const source = useStore(s => s.source);
  const exampleId = useStore(s => s.exampleId);
  const status = useStore(s => s.status);
  const stale = useStore(isStale);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const example = EXAMPLES.find(e => e.id === exampleId);
  const edited = exampleId === '';
  const target = targetNote(options.target);
  const strategy = STRATEGIES.find(s => s.id === options.fusionStrategy) as (typeof STRATEGIES)[number];
  const off = options.disabledPasses.length;
  const backwardMode = BACKWARD_MODES.find(m => m.id === options.backward) as (typeof BACKWARD_MODES)[number];
  const summary = [
    target.label,
    options.backward === 'off' ? null : backwardMode.label,
    ...TOGGLES.filter(toggle => options[toggle.key] as boolean).map(toggle => toggle.label),
    options.verify === 'each-pass' ? null : `verify ${options.verify}`,
    off > 0 ? `${plural(off, 'pass', 'es')} off` : null,
  ].filter(part => part !== null).join(' · ');

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

      {stale && (
        <p className="stale-note" role="status">
          You changed something since the last compile — press Run to update what is on screen.
        </p>
      )}

      <button
        className="options-toggle"
        aria-controls="compile-options"
        aria-expanded={optionsOpen}
        onClick={() => setOptionsOpen(open => !open)}
      >
        <span>{summary}</span>
        <em>{optionsOpen ? 'hide options ▴' : 'options ▾'}</em>
      </button>

      <div id="compile-options" className={optionsOpen ? 'control-more open' : 'control-more'}>
        <p className="blurb">
          {edited
            ? 'Your own code. End it with run(model, inputs) and press Run.'
            : (example as (typeof EXAMPLES)[number]).blurb}
        </p>

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
            <span>Direction</span>
            <select
              value={options.backward}
              onChange={e => actions.setOptions({ backward: e.target.value as BackwardMode })}
            >
              {BACKWARD_MODES.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
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
          <label className="control">
            <span>Verify</span>
            <select
              value={options.verify}
              onChange={e => actions.setOptions({ verify: e.target.value as VerifyLevelName })}
            >
              {VERIFY_LEVELS.map(level => <option key={level.id} value={level.id}>{level.label}</option>)}
            </select>
          </label>
        </div>

        <dl className="control-notes">
          {[
            { term: 'target', note: target.note },
            { term: 'direction', note: backwardMode.note },
            { term: 'fusion', note: options.fusion ? strategy.note : 'fusion is off' },
            { term: 'verify', note: verifyNote(options.verify) },
          ].map(entry => (
            <div key={entry.term}>
              <dt>{entry.term}</dt>
              <dd>{entry.note}</dd>
            </div>
          ))}
        </dl>

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

        <div className="export-row">
          <button
            className="export"
            onClick={() => download('repro.mjs', reproScript(source, options))}
          >
            export repro.mjs
          </button>
          <span>
            a standalone node script that compiles exactly this — the way a bug found here gets re-run where CUDA lives
          </span>
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
      </div>
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
