import type { TraceLog } from '../pipeline/trace.js';

export type ExplainData = Record<string, unknown>;

export type Explain = (subject: string, decision: string, reason: string | null, data?: ExplainData) => void;

export function explainer(trace: TraceLog | null | undefined, category: string): Explain | null {
  if (!trace || !trace.explainsEnabled) return null;
  return (subject, decision, reason, data) => {
    trace.explain(category, subject, decision, reason, data);
  };
}
