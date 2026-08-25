import { IRBuilder } from 'mlfw/compiler/ir/graph/builder.js';

export const MODEL_SOURCE_URL = 'mlfw-model.js';

const FRAME = new RegExp(`${MODEL_SOURCE_URL.replace(/[.]/g, '[.]')}:([0-9]+):`);

const STACK_DEPTH = 80;

type Inserted = { id: number };
type Insertion = (op: Inserted) => Inserted;
type Patchable = { _insert: Insertion };

export function lineFromStack(stack: string | undefined, baseLine: number): number | null {
  if (!stack) return null;
  const match = FRAME.exec(stack);
  if (!match) return null;
  const line = Number(match[1]) - baseLine;
  return line > 0 ? line : null;
}

export type SourceLineRecorder = {
  lines: Map<number, number>;
  stop: () => void;
};

export function recordSourceLines(baseLine: number): SourceLineRecorder {
  const lines = new Map<number, number>();
  const proto = IRBuilder.prototype as unknown as Patchable;
  const original = proto._insert;
  const limits = Error as unknown as { stackTraceLimit?: number };
  const originalLimit = limits.stackTraceLimit;
  limits.stackTraceLimit = STACK_DEPTH;
  let active = true;

  proto._insert = function patched(this: unknown, op: Inserted): Inserted {
    if (active && !lines.has(op.id)) {
      const line = lineFromStack(new Error().stack, baseLine);
      if (line !== null) lines.set(op.id, line);
    }
    return original.call(this, op);
  };

  return {
    lines,
    stop: () => {
      if (!active) return;
      active = false;
      proto._insert = original;
      limits.stackTraceLimit = originalLimit;
    },
  };
}
