import { useMemo, useState } from 'react';
import { actions, useStore } from '../store.js';
import { passLabel } from '../catalog/naming.js';
import type { CompileResponse, TraceEventLite } from '../protocol.js';

const PAGE = 300;
const SKIP_KEYS = new Set(['type', 'level', 'timestamp']);

type Emitter = { step: number; pass: string };
type Row = { index: number; event: TraceEventLite; at: number; text: string; from: Emitter | null };

function payload(event: TraceEventLite): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(event)) {
    if (SKIP_KEYS.has(key) || value === null || value === undefined) continue;
    parts.push(`${key}=${format(value)}`);
  }
  return parts.join(' ');
}

function format(value: unknown): string {
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (Array.isArray(value)) return `[${value.length}]`;
  if (typeof value === 'object') return '{…}';
  return String(value);
}

function emitters(response: CompileResponse): Map<TraceEventLite, Emitter> {
  const owner = new Map<TraceEventLite, Emitter>();
  for (const step of response.steps) {
    for (const event of step.events) {
      if (!owner.has(event)) owner.set(event, { step: step.index, pass: step.pass });
    }
  }
  return owner;
}

function rowsOf(response: CompileResponse): Row[] {
  const start = response.events.length > 0 ? Number(response.events[0].timestamp ?? 0) : 0;
  const owner = emitters(response);
  return response.events.map((event, index) => ({
    index,
    event,
    at: Number(event.timestamp ?? 0) - start,
    text: `${event.type} ${payload(event)}`.toLowerCase(),
    from: owner.get(event) ?? null,
  }));
}

export function TracePanel() {
  const result = useStore(s => s.result);
  const [query, setQuery] = useState('');
  const [type, setType] = useState('');

  const rows = useMemo(() => (result ? rowsOf(result) : []), [result]);
  const types = useMemo(() => [...new Set(rows.map(row => row.event.type))].sort(), [rows]);

  if (!result) {
    return <div className="pane-empty">Run a compile and every trace event the compiler emitted lands here, unfiltered.</div>;
  }

  const needle = query.trim().toLowerCase();
  const matched = rows.filter(row =>
    (type === '' || row.event.type === type) && (needle === '' || row.text.includes(needle)));
  const shown = matched.slice(0, PAGE);

  return (
    <div className="trace">
      <div className="trace-controls">
        <label className="control">
          <span>Type</span>
          <select value={type} onChange={event => setType(event.target.value)}>
            <option value="">all {rows.length}</option>
            {types.map(name => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
        <label className="control grow">
          <span>Filter</span>
          <input
            type="search"
            value={query}
            placeholder="buffer name, pass name, reason…"
            onChange={event => setQuery(event.target.value)}
          />
        </label>
        <span className="trace-count">
          {matched.length} of {rows.length}
          {matched.length > shown.length && ` · showing the first ${PAGE}`}
        </span>
      </div>

      <div className="trace-rows">
        {shown.map(row => <TraceRow key={row.index} row={row} />)}
        {shown.length === 0 && <p className="pane-empty">Nothing matches that filter.</p>}
      </div>
    </div>
  );
}

function TraceRow({ row }: { row: Row }) {
  const body = (
    <>
      <span className="trace-at">{row.at.toFixed(1)}</span>
      <span className="trace-type">{row.event.type}</span>
      <span className="trace-payload">{payload(row.event)}</span>
    </>
  );

  const from = row.from;
  if (from === null) return <div className={`trace-row ${row.event.type}`}>{body}</div>;

  return (
    <button
      className={`trace-row from-pass ${row.event.type}`}
      title={`emitted by ${passLabel(from.pass)} — open the pass that recorded it`}
      onClick={() => actions.reveal(from.step, 'why')}
    >
      {body}
    </button>
  );
}
