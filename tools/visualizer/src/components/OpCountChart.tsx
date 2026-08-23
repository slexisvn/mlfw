import { actions } from '../store.js';
import type { CompileStep } from '../protocol.js';

const HEIGHT = 46;

export function OpCountChart({ steps, selected }: { steps: readonly CompileStep[]; selected: number }) {
  const counts = steps.map(step => step.after.ops);
  const peak = Math.max(...counts, 1);
  const width = Math.max(steps.length, 1);

  return (
    <svg className="opchart" viewBox={`0 0 ${width} ${HEIGHT}`} preserveAspectRatio="none" role="img">
      {steps.map((step, i) => {
        const height = (counts[i] / peak) * (HEIGHT - 4);
        return (
          <rect
            key={step.index}
            x={i}
            y={HEIGHT - height}
            width={0.9}
            height={height}
            className={[step.level, step.index === selected ? 'selected' : '', step.outcome].join(' ')}
            onClick={() => actions.select(step.index)}
          >
            <title>{`${step.pass}: ${step.before.ops} → ${step.after.ops}`}</title>
          </rect>
        );
      })}
    </svg>
  );
}
