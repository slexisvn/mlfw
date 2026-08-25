import { useEffect } from 'react';
import { actions, SPEEDS, useStore, visibleSteps } from '../store.js';

const DWELL_MS = 1500;
const INSTANT_DWELL_MS = 400;

export function Playbar() {
  const steps = useStore(visibleSteps);
  const selected = useStore(s => s.selected);
  const playing = useStore(s => s.playing);
  const speed = useStore(s => s.speed);
  const total = steps.length;
  const at = Math.max(0, steps.findIndex(step => step.index === selected));

  useEffect(() => {
    if (!playing || total === 0) return;
    const dwell = speed === 0 ? INSTANT_DWELL_MS : DWELL_MS / speed;
    const timer = setTimeout(() => {
      if (at >= total - 1) actions.stopPlay();
      else actions.step(1);
    }, dwell);
    return () => clearTimeout(timer);
  }, [playing, at, total, speed]);

  if (total === 0) return null;

  return (
    <div className="playbar">
      <button
        onClick={() => actions.step(-1)}
        disabled={at === 0}
        aria-label="previous step"
        title="previous step (↑ or k)"
      >
        ❮
      </button>
      <button
        className="play"
        onClick={() => actions.togglePlay()}
        aria-label={playing ? 'pause' : 'play the whole compile'}
        title={playing ? 'pause (space)' : 'play the whole compile (space)'}
      >
        {playing ? '❚❚' : '▶'}
      </button>
      <button
        onClick={() => actions.step(1)}
        disabled={at >= total - 1}
        aria-label="next step"
        title="next step (↓ or j)"
      >
        ❯
      </button>

      <input
        className="scrub"
        type="range"
        min={0}
        max={total - 1}
        value={at}
        aria-label="step through the compile"
        onChange={e => actions.select(steps[Number(e.target.value)].index)}
      />

      <span className="counter">step {at + 1}/{total}</span>

      <select
        value={speed}
        aria-label="animation speed"
        title="animation speed"
        onChange={e => actions.setSpeed(Number(e.target.value))}
      >
        {SPEEDS.map(s => <option key={s} value={s}>{s === 0 ? 'instant' : `${s}×`}</option>)}
      </select>
    </div>
  );
}
