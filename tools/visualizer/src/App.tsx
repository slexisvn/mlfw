import { useEffect } from 'react';
import { Controls } from './components/Controls.js';
import { PassTimeline } from './components/PassTimeline.js';
import { SourceEditor } from './components/SourceEditor.js';
import { PaneSwitch } from './components/PaneSwitch.js';
import { StageTabs } from './components/StageTabs.js';
import { actions, useStore } from './store.js';

export function App() {
  const status = useStore(s => s.status);
  const result = useStore(s => s.result);
  const pane = useStore(s => s.pane);

  useEffect(() => { void actions.init(); }, []);

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
          {result && !result.ok && (
            <div className="error">
              <strong>{result.errorPhase ? `failed in ${result.errorPhase}` : 'compile failed'}</strong>
              <pre>{result.error}</pre>
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
