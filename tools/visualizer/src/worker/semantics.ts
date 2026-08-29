import { interpret, Unsupported } from './interpreter.js';
import type { Trace } from './interpreter.js';
import { agree, plural } from '../catalog/naming.js';
import type { CellDiff, SemanticReport } from '../protocol.js';

const SAMPLE_LIMIT = 8;
const EQUAL_EPSILON = 1e-9;

type Named = { name: string; body?: unknown };

function same(a: number, b: number): boolean {
  if (Object.is(a, b)) return true;
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  const scale = Math.max(Math.abs(a), Math.abs(b), 1);
  return Math.abs(a - b) <= EQUAL_EPSILON * scale;
}

function shared(before: Trace, after: Trace): Set<string> {
  const both = new Set<string>();
  for (const buffer of before.buffers) {
    if (after.buffers.has(buffer)) both.add(buffer);
  }
  return both;
}

function bufferOf(cell: string): string {
  return cell.slice(0, cell.indexOf('['));
}

function compareCells(before: Trace, after: Trace, buffers: ReadonlySet<string>) {
  const changed: CellDiff[] = [];
  const dropped: string[] = [];
  const added: string[] = [];
  let compared = 0;

  for (const [cell, value] of before.cells) {
    if (!buffers.has(bufferOf(cell))) continue;
    const now = after.cells.get(cell);
    if (now === undefined) {
      dropped.push(cell);
      continue;
    }
    compared++;
    if (!same(value, now)) changed.push({ cell, before: value, after: now });
  }

  for (const cell of after.cells.keys()) {
    if (!buffers.has(bufferOf(cell))) continue;
    if (!before.cells.has(cell)) added.push(cell);
  }

  return { changed, dropped, added, compared };
}

function verdictFor(report: Omit<SemanticReport, 'verdict'>): string {
  if (!report.ran) return report.reason ?? 'the pass could not be interpreted';

  const stores = plural(report.compared, 'cell');

  if (report.truncated) {
    return `Inconclusive: the interpreter hit its budget before either side finished, so neither store map is `
      + `the program's final state. ${report.changedCount} of ${report.compared} cells differ, but a difference `
      + `here can just as well be one side stopping earlier than the other. Shrink the model and ask again.`;
  }

  if (report.storageReused) {
    return `${plural(report.vanishedBuffers.length, 'buffer')} disappeared into others, so this pass reuses storage. `
      + `A slot that gets reused is meant to end up holding something else once its first reader is done, `
      + `which makes comparing final contents unsound here: ${report.changedCount} of ${report.compared} cells differ `
      + `and this check cannot tell reuse from a miscompile. Bisect or the Result tab decides that one.`;
  }

  if (report.changedCount > 0) {
    const first = report.changed[0];
    return `This pass changed what the program computes: ${first.cell} was ${first.before} and is now ${first.after}`
      + `${report.changedCount > 1 ? `, and ${plural(report.changedCount - 1, 'more cell')} moved with it` : ''}.`;
  }

  if (report.droppedCount > 0 || report.addedCount > 0) {
    return `Every cell both sides write agrees, but the set of cells changed: ${report.droppedCount} no longer written, `
      + `${report.addedCount} newly written. That is a real change unless those cells are dead.`;
  }

  if (report.newBuffers.length > 0) {
    return `${stores} agree exactly, and ${plural(report.newBuffers.length, 'buffer')} ${agree(report.newBuffers.length, 'is')} new — `
      + `a temporary this pass introduced, which nothing downstream of it has read yet.`;
  }

  if (report.reordered) {
    return `${stores} agree exactly, but they are written in a different order. `
      + `For a sequential program that is a schedule change, not a meaning change.`;
  }

  return `${stores} agree exactly, written in the same order. This pass preserved the program's meaning.`;
}

function traceOf(funcs: readonly Named[]): Trace {
  const merged: Trace = {
    cells: new Map(), buffers: new Set(), orderHash: 0, stores: 0, truncated: false,
  };

  for (const func of funcs) {
    const trace = interpret(func.body ?? func);
    for (const [cell, value] of trace.cells) merged.cells.set(cell, value);
    for (const buffer of trace.buffers) merged.buffers.add(buffer);
    merged.orderHash = (merged.orderHash * 31 + trace.orderHash) >>> 0;
    merged.stores += trace.stores;
    merged.truncated = merged.truncated || trace.truncated;
  }

  return merged;
}

export function semanticReport(before: readonly Named[], after: readonly Named[]): SemanticReport {
  const empty = {
    ran: false, reason: null as string | null, truncated: false,
    storesBefore: 0, storesAfter: 0, compared: 0,
    changed: [] as CellDiff[], dropped: [] as string[], added: [] as string[],
    changedCount: 0, droppedCount: 0, addedCount: 0,
    vanishedBuffers: [] as string[], newBuffers: [] as string[],
    storageReused: false, reordered: false,
  };

  let beforeTrace: Trace;
  let afterTrace: Trace;
  try {
    beforeTrace = traceOf(before);
    afterTrace = traceOf(after);
  } catch (error) {
    const reason = error instanceof Unsupported
      ? error.message
      : `the interpreter threw: ${error instanceof Error ? error.message : String(error)}`;
    return { ...empty, reason, verdict: reason };
  }

  const buffers = shared(beforeTrace, afterTrace);
  const { changed, dropped, added, compared } = compareCells(beforeTrace, afterTrace, buffers);
  const vanishedBuffers = [...beforeTrace.buffers].filter(name => !buffers.has(name));
  const newBuffers = [...afterTrace.buffers].filter(name => !buffers.has(name));

  const report = {
    ...empty,
    ran: true,
    truncated: beforeTrace.truncated || afterTrace.truncated,
    storesBefore: beforeTrace.stores,
    storesAfter: afterTrace.stores,
    compared,
    changed: changed.slice(0, SAMPLE_LIMIT),
    dropped: dropped.slice(0, SAMPLE_LIMIT),
    added: added.slice(0, SAMPLE_LIMIT),
    changedCount: changed.length,
    droppedCount: dropped.length,
    addedCount: added.length,
    vanishedBuffers,
    newBuffers,
    storageReused: vanishedBuffers.length > 0,
    reordered: beforeTrace.orderHash !== afterTrace.orderHash,
  };

  return { ...report, verdict: verdictFor(report) };
}
