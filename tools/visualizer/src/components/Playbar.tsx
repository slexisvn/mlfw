import { useEffect } from 'react';
import { actions, SPEEDS, useStore, visibleSteps } from '../store.js';

const DWELL_MS = 1500;

export function Playbar() {
  const steps = useStore(visibleSteps);
  const selected = useStore(s => s.selected);
  const playing = useStore(s => s.playing);
  const speed = useStore(s => s.speed);
  const total = steps.length;
  const at = Math.max(0, steps.findIndex(step => step.index === selected));

  useEffect(() => {
    if (!playing || total === 0) return;
    const dwell = speed === 0 ? 400 : DWELL_MS / speed;
    const timer = setTimeout(() => {
      if (at >= total - 1) actions.stopPlay();
      else actions.step(1);
    }, dwell);
    return () => clearTimeout(timer);
  }, [playing, at, total, speed]);

  if (total === 0) return null;

  return (
    <div className="playbar">
      <button onClick={() => actions.step(-1)} disabled={at === 0} title="previous pass">◀</button>
      <button className="play" onClick={() => actions.togglePlay()} title="play the whole compile">
        {playing ? '❚❚' : '▶'}
      </button>
      <button onClick={() => actions.step(1)} disabled={at >= total - 1} title="next pass">▶</button>

      <input
        className="scrub"
        type="range"
        min={0}
        max={total - 1}
        value={at}
        onChange={e => actions.select(steps[Number(e.target.value)].index)}
      />

      <span className="counter">{at + 1}/{total}</span>

      <select value={speed} onChange={e => actions.setSpeed(Number(e.target.value))} title="animation speed">
        {SPEEDS.map(s => <option key={s} value={s}>{s === 0 ? 'instant' : `${s}×`}</option>)}
      </select>
    </div>
  );
}
