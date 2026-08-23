import { actions, useStore } from '../store.js';
import { IRDiff } from './IRDiff.js';
import { GraphView } from './GraphView.js';
import { OutputPanel } from './OutputPanel.js';
import { ResultPanel } from './ResultPanel.js';
import { Playbar } from './Playbar.js';
import { WhyPanel } from './WhyPanel.js';
import type { StageTab } from '../store.js';

const TABS: { id: StageTab; label: string }[] = [
  { id: 'ir', label: 'IR' },
  { id: 'graph', label: 'Graph' },
  { id: 'why', label: 'Why' },
  { id: 'output', label: 'Output' },
  { id: 'result', label: 'Result' },
];

export function StageTabs() {
  const result = useStore(s => s.result);
  const selected = useStore(s => s.selected);
  const tab = useStore(s => s.tab);
  const step = result ? result.steps[selected] : undefined;

  return (
    <section className="stage">
      <nav className="tabs">
        {TABS.map(t => (
          <button key={t.id} className={t.id === tab ? 'active' : ''} onClick={() => actions.setTab(t.id)}>
            {t.label}
          </button>
        ))}
        {step && (
          <span className="stage-caption">
            {step.pass} · {step.phase} · {step.outcome}
          </span>
        )}
      </nav>

      <Playbar />

      <div className="stage-body">
        {tab === 'output' ? <OutputPanel kernels={result ? result.kernels : []} />
          : tab === 'result' ? <ResultPanel run={result ? result.run : null} />
          : !step ? <div className="pane-empty">Run a compile to inspect it pass by pass.</div>
          : tab === 'ir' ? <IRDiff step={step} />
          : tab === 'graph' ? <GraphView step={step} />
          : <WhyPanel step={step} />}
      </div>
    </section>
  );
}
