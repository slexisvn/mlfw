import { useEffect } from 'react';
import { actions, useStore, visibleSteps } from '../store.js';
import { OpCountChart } from './OpCountChart.js';
import type { CompileStep } from '../protocol.js';

const OUTCOME_MARK: Record<CompileStep['outcome'], string> = {
  changed: '●',
  unchanged: '○',
  failed: '✕',
  unreported: '◐',
};

const LEVEL_LABEL: Record<string, string> = {
  'graph-module': 'graph',
  'graph-func': 'graph',
  tir: 'tir',
  lir: 'lir',
};

export function PassTimeline() {
  const result = useStore(s => s.result);
  const selected = useStore(s => s.selected);
  const disabled = useStore(s => s.options.disabledPasses);
  const onlyChanged = useStore(s => s.onlyChanged);
  const steps = useStore(visibleSteps);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
      if (event.key === 'ArrowDown' || event.key === 'j') actions.step(1);
      else if (event.key === 'ArrowUp' || event.key === 'k') actions.step(-1);
      else return;
      event.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected]);

  if (!result || result.steps.length === 0) {
    return (
      <aside className="timeline empty">
        <p>No compile yet. Press <kbd>Run</kbd> to walk the pipeline.</p>
      </aside>
    );
  }

  const groups = groupByPhase(steps);
  const hidden = result.steps.length - steps.length;

  return (
    <aside className="timeline">
      <OpCountChart steps={steps} selected={selected} />
      <div className="timeline-scroll">
        {groups.map(group => (
          <section key={`${group.phase}-${group.steps[0].index}`}>
            <header className="phase">
              <span className="phase-name">{group.phase}</span>
              <span className="phase-level">{LEVEL_LABEL[group.steps[0].level]}</span>
            </header>
            {group.steps.map(step => (
              <Row
                key={step.index}
                step={step}
                active={step.index === selected}
                off={disabled.includes(step.pass)}
              />
            ))}
          </section>
        ))}
      </div>
      <footer className="timeline-foot">
        <span>
          {steps.length} of {result.steps.length} pass runs changed something · {result.totalMs.toFixed(0)}ms
        </span>
        <button onClick={() => actions.toggleOnlyChanged()}>
          {onlyChanged ? `show all ${result.steps.length}` : `hide the ${hidden === 0 ? 'quiet' : hidden} quiet ones`}
        </button>
      </footer>
    </aside>
  );
}

function Row({ step, active, off }: { step: CompileStep; active: boolean; off: boolean }) {
  const delta = step.after.ops - step.before.ops;

  return (
    <div
      className={['step', active ? 'active' : '', step.outcome, off ? 'off' : ''].filter(Boolean).join(' ')}
      onClick={() => actions.select(step.index)}
    >
      <span className="mark">{step.kind === 'pass' ? OUTCOME_MARK[step.outcome] : step.kind === 'input' ? '▸' : '⇣'}</span>
      <span className="pass-name">{step.pass}</span>
      <span className="ops">
        {step.before.ops}
        {delta !== 0 && <em>→{step.after.ops}</em>}
      </span>
      <span className="ms">{step.kind === 'pass' ? step.durationMs.toFixed(1) : ''}</span>
      {step.kind !== 'pass' ? <span /> : (
        <button
          className="skip"
          title={off ? 'run this pass again' : 'turn this pass off and recompile'}
          onClick={event => { event.stopPropagation(); actions.togglePass(step.pass); }}
        >
          {off ? '↺' : '⊘'}
        </button>
      )}
    </div>
  );
}

function groupByPhase(steps: readonly CompileStep[]): { phase: string; steps: CompileStep[] }[] {
  const groups: { phase: string; steps: CompileStep[] }[] = [];
  for (const step of steps) {
    const last = groups[groups.length - 1];
    if (last && last.phase === step.phase) last.steps.push(step);
    else groups.push({ phase: step.phase, steps: [step] });
  }
  return groups;
}
