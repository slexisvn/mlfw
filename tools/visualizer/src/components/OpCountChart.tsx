import { actions } from '../store.js';
import { levelLabel } from '../catalog/naming.js';
import type { CompileStep } from '../protocol.js';

const HEIGHT = 46;
const FLOOR = 3;
const BAR_WIDTH = 0.86;

export function OpCountChart({ steps, selected }: { steps: readonly CompileStep[]; selected: number }) {
  if (steps.length === 0) return null;

  const peaks = new Map<string, number>();
  for (const step of steps) {
    peaks.set(step.level, Math.max(peaks.get(step.level) ?? 1, step.after.ops));
  }

  const width = steps.length;
  const levels = [...peaks.keys()];

  return (
    <div className="opchart-host">
      <p className="opchart-caption">
        how many nodes the program has after each step
        <span>{levels.length > 1 ? 'scaled inside each IR level' : ''}</span>
      </p>
      <svg
        className="opchart"
        viewBox={`0 0 ${width} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`node count after each of the ${steps.length} steps, scaled inside each IR level`}
      >
        {steps.map((step, i) => {
          const peak = peaks.get(step.level) as number;
          const height = FLOOR + (step.after.ops / peak) * (HEIGHT - FLOOR - 2);
          const boundary = i > 0 && steps[i - 1].level !== step.level;
          return (
            <g key={step.index}>
              {boundary && <rect className="divider" x={i - 0.07} y={0} width={0.14} height={HEIGHT} />}
              <rect
                x={i}
                y={HEIGHT - height}
                width={BAR_WIDTH}
                height={height}
                className={[step.level, step.index === selected ? 'selected' : '', step.outcome].join(' ')}
              />
              <rect
                className="hit"
                x={i}
                y={0}
                width={1}
                height={HEIGHT}
                onClick={() => actions.select(step.index)}
              >
                <title>
                  {`${step.pass} · ${levelLabel(step.level)} · ${step.before.ops} → ${step.after.ops} nodes`}
                </title>
              </rect>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
