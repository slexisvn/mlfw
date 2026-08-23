import { chapterUrl, passNote } from '../catalog/passes.js';
import type { CompileStep, TraceEventLite } from '../protocol.js';

const INTERESTING = new Set(['explain', 'pass_detail', 'warning', 'memory', 'codegen', 'error']);

export function WhyPanel({ step }: { step: CompileStep }) {
  const events = step.events.filter(event => INTERESTING.has(event.type));

  return (
    <div className="why">
      <PassCard step={step} />
      {events.length === 0
        ? (step.kind === 'input' ? null : (
          <p className="why-quiet">
            This run emitted no decisions. Passes explain themselves through the trace log, and this one
            had nothing to report for this graph.
          </p>
        ))
        : events.map((event, i) => <EventCard key={i} event={event} />)}
    </div>
  );
}

function PassCard({ step }: { step: CompileStep }) {
  const note = passNote(step.pass);
  const delta = step.after.ops - step.before.ops;

  if (step.kind === 'lowering' && step.phase === 'codegen') {
    return (
      <article className="why-card pass">
        <header>
          <span className="tag">codegen</span>
          <span className="subject">{step.pass}</span>
        </header>
        <p className="decision">
          The last translation. Every loop, buffer and store becomes text in the target language —
          {' '}{step.before.ops} IR nodes emitted {step.after.ops} lines of source.
        </p>
        <p className="reason">
          Here the trail survives: codegen writes the loop variables straight into the code, so a loop in
          the IR can be matched to the loop it produced by name. Nothing is guessed.
        </p>
      </article>
    );
  }

  if (step.kind === 'lowering') {
    return (
      <article className="why-card pass">
        <header>
          <span className="tag">lowering</span>
          <span className="subject">{step.pass}</span>
        </header>
        <p className="decision">
          Not a pass — a change of language. Every op on the left becomes a loop nest on the right, so the
          node count stops meaning what it meant: {step.before.ops} ops became {step.after.ops} IR nodes.
        </p>
        <p className="reason">
          An op is linked to its loop nest when the block still carries its name — lowering builds blocks
          called <code>add_block_5</code>, <code>fusion_block_2</code>. Ops whose rule names the block after
          the math instead of the op (<code>dot</code> becomes <code>matmul_init</code> and
          <code>matmul_acc</code>) cannot be linked, so they dissolve and their loops appear instead. That
          gap is real: past this boundary the IR no longer remembers which op it came from.
        </p>
      </article>
    );
  }

  if (step.kind === 'input') {
    return (
      <article className="why-card pass">
        <header>
          <span className="tag">input</span>
          <span className="subject">traced graph</span>
        </header>
        <p className="decision">
          Your model, recorded as a dataflow graph. Nothing has been optimized yet — {step.after.ops} ops
          exactly as tracing produced them.
        </p>
      </article>
    );
  }

  return (
    <article className="why-card pass">
      <header>
        <span className="tag">pass</span>
        <span className="subject">{step.pass}</span>
      </header>
      {note && <p className="decision">{note.summary}</p>}
      <p className="reason">
        ran in {step.phase} · {step.before.ops} ops in, {step.after.ops} out
        {delta !== 0 && ` (${delta > 0 ? '+' : ''}${delta})`} · {step.durationMs.toFixed(1)}ms
      </p>
      {note && (
        <a className="chapter" href={chapterUrl(note)} target="_blank" rel="noreferrer">
          read {note.chapterTitle} →
        </a>
      )}
    </article>
  );
}

function EventCard({ event }: { event: TraceEventLite }) {
  if (event.type === 'explain') {
    return (
      <article className="why-card explain">
        <header>
          <span className="tag">{String(event.category)}</span>
          <span className="subject">{String(event.subject)}</span>
        </header>
        <p className="decision">{String(event.decision)}</p>
        {event.reason ? <p className="reason">because {String(event.reason)}</p> : null}
        <Extras event={event} skip={['type', 'level', 'timestamp', 'category', 'subject', 'decision', 'reason']} />
      </article>
    );
  }

  if (event.type === 'warning') {
    return (
      <article className="why-card warning">
        <header><span className="tag">warning</span><span className="subject">{String(event.phase)}</span></header>
        <p className="decision">{String(event.message)}</p>
      </article>
    );
  }

  return (
    <article className={`why-card ${event.type}`}>
      <header><span className="tag">{event.type}</span></header>
      <Extras event={event} skip={['type', 'level', 'timestamp']} />
    </article>
  );
}

function Extras({ event, skip }: { event: TraceEventLite; skip: readonly string[] }) {
  const entries = Object.entries(event).filter(([key, value]) =>
    !skip.includes(key) && value !== null && value !== undefined);
  if (entries.length === 0) return null;

  return (
    <dl className="extras">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{format(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function format(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}
