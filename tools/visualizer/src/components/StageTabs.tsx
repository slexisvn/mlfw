import { useLayoutEffect, useRef } from 'react';
import { actions, useStore } from '../store.js';
import { TAB_NOTES } from '../catalog/glossary.js';
import { phaseLabel } from '../catalog/naming.js';
import { IRDiff } from './IRDiff.js';
import { GraphView } from './GraphView.js';
import { OutputPanel } from './OutputPanel.js';
import { ResultPanel } from './ResultPanel.js';
import { HealthPanel } from './HealthPanel.js';
import { ComparePanel } from './ComparePanel.js';
import { BisectPanel } from './BisectPanel.js';
import { SemanticsPanel } from './SemanticsPanel.js';
import { ProfilePanel } from './ProfilePanel.js';
import { TracePanel } from './TracePanel.js';
import { TuningPanel } from './TuningPanel.js';
import { MemoryPanel } from './MemoryPanel.js';
import { Playbar } from './Playbar.js';
import { WhyPanel } from './WhyPanel.js';
import type { CompileStep } from '../protocol.js';

function captionFor(step: CompileStep): string {
  if (step.kind === 'input') return 'your model, straight out of tracing';
  if (step.kind === 'lowering') return `${step.pass} · not a pass, a translation`;
  if (step.kind === 'primitive') return `${step.pass} · one schedule primitive on ${step.parent ?? 'the loop nest'}`;
  return `${step.pass} · ${phaseLabel(step.phase)} · ${step.outcome}`;
}

export function StageTabs() {
  const result = useStore(s => s.result);
  const selected = useStore(s => s.selected);
  const tab = useStore(s => s.tab);
  const step = result ? result.steps[selected] : undefined;
  const caption = step ? captionFor(step) : '';
  const activeTab = useRef<HTMLButtonElement>(null);

  useLayoutEffect(() => {
    activeTab.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [tab, caption]);

  return (
    <section className="stage">
      <nav className="tabs">
        <div className="tab-strip" role="tablist">
          {TAB_NOTES.map(t => (
            <button
              key={t.id}
              role="tab"
              ref={t.id === tab ? activeTab : undefined}
              className={t.id === tab ? 'active' : ''}
              aria-selected={t.id === tab}
              title={t.meaning}
              onClick={() => actions.setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        {step && <span className="stage-caption" title={caption}>{caption}</span>}
      </nav>

      <Playbar />

      <div className="stage-body">
        {tab === 'tuning' ? <TuningPanel rounds={result ? result.tuningRounds : []} />
          : tab === 'compare' ? <ComparePanel />
          : tab === 'bisect' ? <BisectPanel />
          : tab === 'profile' ? <ProfilePanel />
          : tab === 'trace' ? <TracePanel />
          : tab === 'memory' ? <MemoryPanel plans={result ? result.memoryPlans : []} />
          : tab === 'output' ? <OutputPanel kernels={result ? result.kernels : []} reports={result ? result.kernelReports : []} />
          : tab === 'result' ? <ResultPanel run={result ? result.run : null} />
          : tab === 'health' ? <HealthPanel run={result ? result.run : null} />
          : !step ? <div className="pane-empty">Run a compile and every step becomes inspectable here.</div>
          : tab === 'ir' ? <IRDiff step={step} />
          : tab === 'graph' ? <GraphView step={step} />
          : tab === 'semantics' ? <SemanticsPanel step={step} />
          : <WhyPanel step={step} />}
      </div>
    </section>
  );
}
