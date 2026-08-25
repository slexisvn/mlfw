import { actions, useStore } from '../store.js';
import { TAB_NOTES } from '../catalog/glossary.js';
import { phaseLabel } from '../catalog/naming.js';
import { IRDiff } from './IRDiff.js';
import { GraphView } from './GraphView.js';
import { OutputPanel } from './OutputPanel.js';
import { ResultPanel } from './ResultPanel.js';
import { Playbar } from './Playbar.js';
import { WhyPanel } from './WhyPanel.js';
import type { StageTab } from '../store.js';
import type { CompileStep } from '../protocol.js';

const TABS: { id: StageTab; label: string }[] = [
  { id: 'ir', label: 'IR' },
  { id: 'graph', label: 'Graph' },
  { id: 'why', label: 'Why' },
  { id: 'output', label: 'Output' },
  { id: 'result', label: 'Result' },
];

const NOTE_BY_TAB = new Map(TAB_NOTES.map(note => [note.tab, note.meaning]));

function captionFor(step: CompileStep): string {
  if (step.kind === 'input') return 'your model, straight out of tracing';
  if (step.kind === 'lowering') return `${step.pass} · not a pass, a translation`;
  return `${step.pass} · ${phaseLabel(step.phase)} · ${step.outcome}`;
}

export function StageTabs() {
  const result = useStore(s => s.result);
  const selected = useStore(s => s.selected);
  const tab = useStore(s => s.tab);
  const step = result ? result.steps[selected] : undefined;

  return (
    <section className="stage">
      <nav className="tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={t.id === tab ? 'active' : ''}
            aria-current={t.id === tab ? 'true' : undefined}
            title={NOTE_BY_TAB.get(t.label)}
            onClick={() => actions.setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
        {step && <span className="stage-caption">{captionFor(step)}</span>}
      </nav>

      <Playbar />

      <div className="stage-body">
        {tab === 'output' ? <OutputPanel kernels={result ? result.kernels : []} />
          : tab === 'result' ? <ResultPanel run={result ? result.run : null} />
          : !step ? <div className="pane-empty">Run a compile and every step becomes inspectable here.</div>
          : tab === 'ir' ? <IRDiff step={step} />
          : tab === 'graph' ? <GraphView step={step} />
          : <WhyPanel step={step} />}
      </div>
    </section>
  );
}
