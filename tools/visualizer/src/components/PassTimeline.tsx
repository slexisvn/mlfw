import { useEffect, useMemo, useRef, useState } from 'react';
import { actions, childCount, disabledPasses, isCollapsed, passRunCount, provenance, useStore, visibleSteps } from '../store.js';
import { markFor } from '../catalog/glossary.js';
import { PRESENCE_MARKS } from '../catalog/provenance.js';
import { levelBadge, levelLabel, passLabel, phaseLabel, plural } from '../catalog/naming.js';
import { OpCountChart } from './OpCountChart.js';
import type { DisabledPass } from '../store.js';
import type { Presence, Provenance } from '../catalog/provenance.js';
import type { CompileStep, IRLevelName } from '../protocol.js';

type Group = { phase: string; unit: string | null; level: IRLevelName | null; steps: CompileStep[]; disabled: DisabledPass[] };

export function PassTimeline() {
  const result = useStore(s => s.result);
  const selected = useStore(s => s.selected);
  const onlyChanged = useStore(s => s.onlyChanged);
  const steps = useStore(visibleSteps);
  const runs = useStore(passRunCount);
  const disabled = useStore(disabledPasses);
  const found = useStore(provenance);
  const activeRow = useRef<HTMLButtonElement>(null);

  const ordinals = useMemo(() => runOrdinals(result ? result.steps : []), [result]);

  useEffect(() => {
    activeRow.current?.scrollIntoView({ block: 'nearest' });
  }, [selected, onlyChanged]);

  if (!result || result.steps.length === 0) {
    return (
      <aside className="timeline empty">
        <p>Nothing compiled yet. Press <kbd>Run</kbd> and every pass the compiler runs shows up here.</p>
      </aside>
    );
  }

  const groups = groupSteps(steps, disabled);
  const units = new Set(result.steps.map(step => step.unit).filter(unit => unit !== null));

  return (
    <aside className="timeline">
      <OpCountChart steps={steps} selected={selected} />
      <FindBar />
      <div className="timeline-scroll">
        {groups.map(group => (
          <section key={`${group.phase}-${group.steps[0]?.index ?? group.phase}`}>
            <header className="phase">
              <span className="phase-name">{phaseLabel(group.phase)}</span>
              {units.size > 1 && group.unit && (
                <span className="phase-unit" title="which function this phase was compiling">
                  {group.unit}
                </span>
              )}
              {group.level && (
                <span className="phase-level" title={`this phase works on the ${levelLabel(group.level)}`}>
                  {levelBadge(group.level)}
                </span>
              )}
            </header>
            {group.steps.map(step => (
              <Row
                key={step.index}
                step={step}
                ordinal={ordinals.get(step.index) ?? null}
                active={step.index === selected}
                presence={found && found.hits > 0 ? found.marks.get(step.index) ?? 'absent' : null}
                rowRef={step.index === selected ? activeRow : undefined}
              />
            ))}
            {group.disabled.map(entry => <OffRow key={entry.name} entry={entry} />)}
          </section>
        ))}
      </div>
      <footer className="timeline-foot">
        <span>
          {runs.changed} of {runs.total} pass runs changed the IR · {result.totalMs.toFixed(0)}ms
        </span>
        <button
          onClick={() => actions.toggleOnlyChanged()}
          title={onlyChanged
            ? 'also list the passes that ran and did nothing'
            : 'list only the passes that rewrote something'}
        >
          {onlyChanged ? `show all ${result.steps.length}` : `hide ${runs.quiet} quiet`}
        </button>
      </footer>
    </aside>
  );
}

const FIND_DEBOUNCE_MS = 140;

function findNote(found: Provenance): string {
  if (found.hits === 0) {
    return found.insideLonger > 0
      ? `never a name on its own — only inside longer ones, in ${plural(found.insideLonger, 'step')}`
      : 'never in the IR';
  }
  return `${found.bornAt === null ? 'there from the start' : `born at ${found.bornAt}`}`
    + `${found.diedAt === null ? ', still there at the end' : `, gone after ${found.diedAt}`}`;
}

function FindBar() {
  const find = useStore(s => s.find);
  const found = useStore(provenance);
  const [draft, setDraft] = useState(find);

  useEffect(() => setDraft(find), [find]);

  useEffect(() => {
    if (draft === find) return;
    const timer = setTimeout(() => actions.setFind(draft), FIND_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [draft, find]);

  return (
    <div className="find-bar">
      <input
        type="search"
        value={draft}
        placeholder="track an op, a value or a buffer — %13, dot, compute_1"
        aria-label="track a name through the pipeline"
        title="Matches a whole name as printed, and falls back to matching inside longer names. Op and buffer names are stable; %n is the printer's numbering and a pass is free to reuse it, so a value can look reborn."
        onChange={event => setDraft(event.target.value)}
      />
      {draft !== '' && (
        <button className="find-clear" aria-label="stop tracking this name" title="clear" onClick={() => setDraft('')}>
          ×
        </button>
      )}
      {found && (
        <span className={found.hits === 0 ? 'find-note empty' : 'find-note'} role="status">
          {findNote(found)}
        </span>
      )}
    </div>
  );
}

function Row(
  { step, ordinal, active, presence, rowRef }: {
    step: CompileStep;
    ordinal: string | null;
    active: boolean;
    presence: Presence | null;
    rowRef?: React.RefObject<HTMLButtonElement | null>;
  },
) {
  const kids = useStore(s => childCount(s, step.index));
  const folded = useStore(s => isCollapsed(s, step.index));
  const delta = step.after.ops - step.before.ops;
  const mark = markFor(step);
  const isPass = step.kind === 'pass';
  const broke = step.verify !== null && step.verify.introduced.length > 0;
  const inherited = step.verify !== null && step.verify.introduced.length === 0 && step.verify.carried.length > 0;

  return (
    <div className={[
      'step-row', active ? 'active' : '', step.outcome, step.kind,
      broke ? 'invalid' : '', presence ? `seen-${presence}` : '',
    ].filter(Boolean).join(' ')}>
      {kids > 0 ? (
        <button
          className="twist"
          aria-expanded={!folded}
          aria-label={folded
            ? `show the ${kids} primitives ${step.pass} applied`
            : `fold the ${kids} primitives ${step.pass} applied away`}
          title={folded
            ? `${kids} schedule primitives are folded in here`
            : 'fold these primitives back into the pass'}
          onClick={() => actions.toggleCollapse(step.index)}
        >
          {folded ? '▸' : '▾'}
        </button>
      ) : (
        <span className="twist" />
      )}
      <button
        ref={rowRef}
        className="step"
        aria-current={active ? 'true' : undefined}
        title={isPass ? `${passLabel(step.pass)} — ${mark.label}` : mark.label}
        onClick={() => actions.select(step.index)}
      >
        <span className="mark" aria-hidden="true">{mark.glyph}</span>
        <span className="seen" title={presence ? PRESENCE_MARKS[presence].label : undefined}>
          {presence ? PRESENCE_MARKS[presence].glyph : ''}
        </span>
        <span className="pass-name">
          {step.pass}
          {broke && (
            <em className="invalid-badge" title={`the IR fails ${plural(step.verify?.introduced.length ?? 0, 'invariant check')} after this pass that it passed before`}>
              invalid
            </em>
          )}
          {inherited && (
            <em className="carried-badge" title="the IR was already failing a verifier check before this pass ran">
              was invalid
            </em>
          )}
          {ordinal && <em className="ordinal">{ordinal}</em>}
          {kids > 0 && folded && <em className="kids">+{kids}</em>}
        </span>
        <span className="ops">
          {step.before.ops}
          {delta !== 0 && <em>→{step.after.ops}</em>}
        </span>
        <span className="ms">{isPass ? step.durationMs.toFixed(1) : ''}</span>
      </button>
      {isPass && (
        <button
          className="skip"
          aria-label={`turn ${step.pass} off and recompile`}
          title="turn this pass off and recompile, to see what it was buying you"
          onClick={() => actions.togglePass(step.pass)}
        >
          ⊘
        </button>
      )}
    </div>
  );
}

const OFF_STATUS: Record<DisabledPass['status'], { label: string; title: string }> = {
  off: { label: 'off', title: 'the compiler confirmed it skipped this pass' },

  ignored: {
    label: 'ignored',
    title: 'you turned this pass off but the compiler ran it anyway — the request never reached the pass manager for this IR level',
  },
  pending: { label: 'pending', title: 'this compile does not reflect the switch yet — press Run' },
};

function OffRow({ entry }: { entry: DisabledPass }) {
  const status = OFF_STATUS[entry.status];

  return (
    <div className={`step-row off off-${entry.status}`}>
      <span className="twist" />
      <span className="step">
        <span className="mark" aria-hidden="true">⊘</span>
        <span className="pass-name">{entry.name}</span>
        <span className="ops" title={status.title}>{status.label}</span>
        <span className="ms" />
      </span>
      <button
        className="skip restore"
        aria-label={`put ${entry.name} back and recompile`}
        title={`put ${passLabel(entry.name)} back and recompile`}
        onClick={() => actions.togglePass(entry.name)}
      >
        ↺
      </button>
    </div>
  );
}

function runOrdinals(steps: readonly CompileStep[]): Map<number, string> {
  const totals = new Map<string, number>();
  for (const step of steps) {
    if (step.kind !== 'pass') continue;
    totals.set(step.pass, (totals.get(step.pass) ?? 0) + 1);
  }

  const seen = new Map<string, number>();
  const ordinals = new Map<number, string>();
  for (const step of steps) {
    if (step.kind !== 'pass') continue;
    const total = totals.get(step.pass) as number;
    if (total < 2) continue;
    const at = (seen.get(step.pass) ?? 0) + 1;
    seen.set(step.pass, at);
    ordinals.set(step.index, `${at}/${total}`);
  }
  return ordinals;
}

function groupSteps(steps: readonly CompileStep[], disabled: readonly DisabledPass[]): Group[] {
  const groups: Group[] = [];
  for (const step of steps) {
    const last = groups[groups.length - 1];
    if (last && last.phase === step.phase && last.unit === step.unit) last.steps.push(step);
    else groups.push({ phase: step.phase, unit: step.unit, level: step.level, steps: [step], disabled: [] });
  }

  const orphans: DisabledPass[] = [];
  for (const entry of disabled) {
    const home = groups.find(group => group.phase === entry.phase);
    if (home) home.disabled.push(entry);
    else orphans.push(entry);
  }

  if (orphans.length > 0) {
    groups.push({ phase: 'turned off', unit: null, level: null, steps: [], disabled: orphans });
  }
  return groups;
}
