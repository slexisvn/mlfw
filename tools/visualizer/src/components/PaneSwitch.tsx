import { actions, useStore } from '../store.js';
import type { Pane } from '../store.js';

const PANES: { id: Pane; label: string }[] = [
  { id: 'source', label: 'Code' },
  { id: 'timeline', label: 'Passes' },
  { id: 'stage', label: 'View' },
];

export function PaneSwitch() {
  const pane = useStore(s => s.pane);
  const steps = useStore(s => (s.result ? s.result.steps.length : 0));

  return (
    <nav className="pane-switch">
      {PANES.map(entry => (
        <button
          key={entry.id}
          className={entry.id === pane ? 'active' : ''}
          onClick={() => actions.setPane(entry.id)}
        >
          {entry.label}
          {entry.id === 'timeline' && steps > 0 && <em>{steps}</em>}
        </button>
      ))}
    </nav>
  );
}
