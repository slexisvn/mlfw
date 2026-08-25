import { useEffect } from 'react';
import { Controls } from './components/Controls.js';
import { PassTimeline } from './components/PassTimeline.js';
import { SourceEditor } from './components/SourceEditor.js';
import { PaneSwitch } from './components/PaneSwitch.js';
import { StageTabs } from './components/StageTabs.js';
import { actions, useStore } from './store.js';
import { passLabel } from './catalog/naming.js';

const TYPING = new Set(['INPUT', 'TEXTAREA']);

function isTyping(target: EventTarget | null): boolean {
  const node = target as HTMLElement | null;
  if (!node) return false;
  return TYPING.has(node.tagName) || node.isContentEditable;
}

export function App() {
  const status = useStore(s => s.status);
  const failure = useStore(s => s.failure);
  const pane = useStore(s => s.pane);

  useEffect(() => { void actions.init(); }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (isTyping(event.target) || event.metaKey || event.ctrlKey || event.altKey) return;

      if (event.key === 'ArrowDown' || event.key === 'j') actions.step(1);
      else if (event.key === 'ArrowUp' || event.key === 'k') actions.step(-1);
      else if (event.key === ' ' && (event.target as HTMLElement | null)?.tagName !== 'BUTTON') actions.togglePlay();
      else return;

      event.preventDefault();
    };

    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <h1>mlfw <span>pass visualizer</span></h1>
        <p className="tagline">a model goes in, every pass shows its work, a kernel comes out</p>
        <PaneSwitch />
      </header>

      <main className={`columns show-${pane}`}>
        <div className="column source">
          <Controls />
          <SourceEditor />
          {failure && (
            <div className="error" role="alert">
              <strong>
                {failure.errorPhase
                  ? `${passLabel(failure.errorPhase)} threw before it finished`
                  : 'that did not compile'}
              </strong>
              <pre>{failure.error}</pre>
              <p>Fix the code and press Run — the last successful compile is still on screen.</p>
            </div>
          )}
        </div>

        <PassTimeline />
        <StageTabs />
      </main>

      {status === 'starting' && <div className="booting">loading the compiler…</div>}
    </div>
  );
}
