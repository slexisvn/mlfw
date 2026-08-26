import { useMemo, useState } from 'react';
import type { BufferLifetime, MemoryPlan } from '../protocol.js';

const ROW_HEIGHT = 16;
const ROW_GAP = 3;
const CURVE_HEIGHT = 90;
const SLOT_TONES = 6;

type Curve = { live: number[]; planned: number[]; livePeak: number };

function liveAt(buffers: readonly BufferLifetime[], step: number): BufferLifetime[] {
  return buffers.filter(b => b.firstUse <= step && step <= b.lastUse);
}

function curveOf(plan: MemoryPlan): Curve {
  const live: number[] = [];
  const planned: number[] = [];
  let livePeak = 0;

  for (let step = 0; step < plan.steps; step++) {
    let sum = 0;
    let high = 0;
    for (const buffer of liveAt(plan.buffers, step)) {
      sum += buffer.bytes;
      high = Math.max(high, buffer.slot + buffer.bytes);
    }
    live.push(sum);
    planned.push(high);
    livePeak = Math.max(livePeak, sum);
  }

  return { live, planned, livePeak };
}

function toneOf(slot: number, slots: readonly number[]): number {
  const index = slots.indexOf(slot);
  return index < 0 ? 0 : index % SLOT_TONES;
}

function bytes(value: number): string {
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

function path(values: readonly number[], scale: number, width: number): string {
  if (values.length === 0) return '';
  const step = width / Math.max(values.length, 1);
  const points = values.map((value, i) => `${(i * step).toFixed(2)},${(CURVE_HEIGHT - value * scale).toFixed(2)}`);
  const last = values[values.length - 1];
  points.push(`${width.toFixed(2)},${(CURVE_HEIGHT - last * scale).toFixed(2)}`);
  return `M${points.join(' L')}`;
}

export function MemoryPanel({ plans }: { plans: readonly MemoryPlan[] }) {
  const [selected, setSelected] = useState(0);
  const plan = plans[Math.min(selected, plans.length - 1)];
  const curve = useMemo(() => (plan ? curveOf(plan) : null), [plan]);

  if (!plan || !curve) {
    return (
      <div className="pane-empty">
        Nothing was planned into memory. Every value in this program stayed in a register, or the
        pipeline stopped before the memory pass ran.
      </div>
    );
  }

  const slots = [...new Set(plan.buffers.map(b => b.slot))].sort((a, b) => a - b);
  const saved = plan.totalBytesIfNeverShared - plan.peakMemory;
  const overhead = plan.peakMemory - curve.livePeak;
  const width = Math.max(plan.steps, 1);
  const ceiling = Math.max(plan.totalBytesIfNeverShared, plan.peakMemory, 1);
  const scale = (CURVE_HEIGHT - 6) / ceiling;

  return (
    <div className="memory">
      {plans.length > 1 && (
        <nav className="memory-funcs">
          {plans.map((candidate, i) => (
            <button
              key={candidate.func}
              className={i === selected ? 'active' : ''}
              onClick={() => setSelected(i)}
            >
              {candidate.func}
            </button>
          ))}
        </nav>
      )}

      <section className="memory-verdict">
        <h2>{bytes(plan.totalBytesIfNeverShared)} of temporaries fit in {bytes(plan.peakMemory)}</h2>
        <p>
          {plan.buffers.length} temporary buffer{plan.buffers.length === 1 ? '' : 's'} live somewhere inside{' '}
          {plan.func}, sharing {slots.length} slot{slots.length === 1 ? '' : 's'} between them. Two buffers can
          share a slot exactly when their lifetimes never touch — the bars below are those lifetimes, and the
          colour is the slot.
          {saved > 0 && ` Reuse saved ${bytes(saved)} against giving every buffer an address of its own.`}
          {' '}At its worst moment {bytes(curve.livePeak)} is genuinely live, and the plan holds {bytes(plan.peakMemory)}
          {overhead > 0
            ? ` — ${bytes(overhead)} of that is space a live buffer is sitting above and cannot be handed out.`
            : ' — nothing is wasted between them.'}
        </p>
      </section>

      <section className="memory-block">
        <h3>bytes held<span>what the plan reserves, against what is genuinely live</span></h3>
        <svg
          className="memory-curve"
          viewBox={`0 0 ${width} ${CURVE_HEIGHT}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`memory held across ${plan.steps} statements, peaking at ${plan.peakMemory} bytes`}
        >
          <path className="curve naive" d={path(curve.live, scale, width)} vectorEffect="non-scaling-stroke" />
          <path className="curve planned" d={path(curve.planned, scale, width)} vectorEffect="non-scaling-stroke" />
          <line
            className="peak-line"
            x1={0}
            x2={width}
            y1={CURVE_HEIGHT - plan.totalBytesIfNeverShared * scale}
            y2={CURVE_HEIGHT - plan.totalBytesIfNeverShared * scale}
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        <p className="memory-legend">
          <span className="key planned" /> reserved by the plan, peaking at {bytes(plan.peakMemory)}
          <span className="key naive" /> genuinely live, peaking at {bytes(curve.livePeak)}
          <span className="key ceiling" /> one address each, {bytes(plan.totalBytesIfNeverShared)}
        </p>
      </section>

      <section className="memory-block">
        <h3>lifetimes<span>one row per buffer, one column per statement</span></h3>
        <div className="memory-rows">
          {plan.buffers.map(buffer => (
            <div className="memory-row" key={buffer.name}>
              <span className="memory-name" title={`${buffer.scope} · slot at byte ${buffer.slot}`}>
                {buffer.name}
              </span>
              <svg
                className="memory-track"
                viewBox={`0 0 ${width} ${ROW_HEIGHT}`}
                preserveAspectRatio="none"
                role="img"
                aria-label={
                  `${buffer.name} is live from statement ${buffer.firstUse} to ${buffer.lastUse}`
                  + ` in ${bytes(buffer.bytes)} at slot ${buffer.slot}`
                }
              >
                <rect className="track" x={0} y={0} width={width} height={ROW_HEIGHT} />
                <rect
                  className={`span tone-${toneOf(buffer.slot, slots)}`}
                  x={buffer.firstUse}
                  y={ROW_GAP}
                  width={buffer.lastUse - buffer.firstUse + 1}
                  height={ROW_HEIGHT - ROW_GAP * 2}
                />
              </svg>
              <span className="memory-size">{bytes(buffer.bytes)}</span>
              <span className="memory-note">
                {buffer.sharesWith
                  ? `reuses the slot ${buffer.sharesWith} just freed`
                  : `slot at byte ${buffer.slot}`}
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
