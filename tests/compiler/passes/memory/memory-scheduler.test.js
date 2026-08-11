import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { TraceLevel } from '../../../../src/compiler/pipeline/trace.js';

const BIG = 4096;
const SMALL = 8;
const t = (shape) => new TensorType(shape, ScalarType.F32);

function widePeaksBeforeReductions() {
  const input = t([2, BIG]);
  return buildFunction('chains', [input, input, input], [t([2])], (b, a) => {
    const wide = a.map((x) => b.mul(x, x).getResult(0));
    const zero = b.scalarConstant(0, ScalarType.F32).getResult(0);
    const reduced = wide.map((x) => b.reduce(x, zero, [1], 'sum').getResult(0));
    b.returnOp([b.add(b.add(reduced[0], reduced[1]).getResult(0), reduced[2]).getResult(0)]);
  });
}

function compileWith(scheduleForPeak) {
  const events = [];
  const result = compileGraph(widePeaksBeforeReductions(), CPUTarget(), {
    memory: { scheduleForPeak },
    fusion: { enabled: false },
    trace: { level: TraceLevel.VERBOSE, sink: (e) => events.push(e) },
  });
  const memory = events.filter((e) => e.type === 'memory' && e.funcName === 'chains');
  return { result, peak: memory.length > 0 ? memory[0].peakMemory : null, events };
}

function run(result) {
  const make = (seed) => Float32Array.from({ length: 2 * BIG }, (_, i) => Math.sin((i + seed) * 0.001));
  const out = new Float32Array(2);
  result.run('chains', make(1), make(2), make(3), out);
  return Array.from(out);
}

describe('memory-aware statement scheduling', () => {
  it('lowers planned peak memory relative to the program order', () => {
    const scheduled = compileWith(true);
    const asWritten = compileWith(false);
    expect(asWritten.peak).not.toBeNull();
    expect(scheduled.peak).toBeLessThan(asWritten.peak);
  });

  it('reports the peak it achieved against the original order', () => {
    const { events } = compileWith(true);
    const scheduling = events.filter((e) => e.type === 'function' && e.phase === 'memoryScheduling');
    expect(scheduling).toHaveLength(1);
    expect(scheduling[0].peakBytes).toBeLessThan(scheduling[0].originalPeakBytes);
  });

  it('produces bit-identical results to the unscheduled program', () => {
    expect(run(compileWith(true).result)).toEqual(run(compileWith(false).result));
  });

  it('leaves the order alone when it cannot improve on it', () => {
    const events = [];
    compileGraph(
      buildFunction('chain', [t([SMALL])], [t([SMALL])], (b, a) => {
        let x = a[0];
        for (let i = 0; i < 4; i++) x = b.add(x, x).getResult(0);
        b.returnOp([x]);
      }),
      CPUTarget(),
      { memory: { scheduleForPeak: true }, fusion: { enabled: false },
        trace: { level: TraceLevel.VERBOSE, sink: (e) => events.push(e) } }
    );
    expect(events.filter((e) => e.type === 'function' && e.phase === 'memoryScheduling')).toHaveLength(0);
  });
});
