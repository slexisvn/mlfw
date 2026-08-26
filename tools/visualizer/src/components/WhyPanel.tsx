import { chapterUrl, passNote } from '../catalog/passes.js';
import { formatMetric } from '../catalog/metrics.js';
import { isHiddenMetric, levelLabel, metricLabel, metricValue, passLabel, phaseLabel } from '../catalog/naming.js';
import type { CompileStep, TraceEventLite } from '../protocol.js';

const INTERESTING = new Set(['explain', 'pass_detail', 'warning', 'memory', 'codegen', 'error']);
const BASE_KEYS = ['type', 'level', 'timestamp'];
const EXPLAIN_KEYS = [...BASE_KEYS, 'category', 'subject', 'decision', 'reason'];
const SUBJECT_LIMIT = 6;

type Group = { event: TraceEventLite; subjects: string[] };

function groupEvents(events: readonly TraceEventLite[]): Group[] {
  const groups: Group[] = [];
  const byKey = new Map<string, Group>();

  for (const event of events) {
    const key = [event.type, event.category, event.decision, event.reason].map(String).join('|');
    const existing = byKey.get(key);
    const subject = event.subject === undefined ? null : String(event.subject);

    if (existing) {
      if (subject) existing.subjects.push(subject);
      continue;
    }

    const group: Group = { event, subjects: subject ? [subject] : [] };
    byKey.set(key, group);
    groups.push(group);
  }

  return groups;
}

export function WhyPanel({ step }: { step: CompileStep }) {
  const events = step.events.filter(event => INTERESTING.has(event.type));
  const groups = groupEvents(events);
  const explains = step.kind !== 'pass';

  return (
    <div className="why">
      <PassCard step={step} />
      {groups.length === 0
        ? (explains ? null : <QuietNote step={step} />)
        : groups.map((group, i) => <EventCard key={i} group={group} />)}
    </div>
  );
}

function QuietNote({ step }: { step: CompileStep }) {
  const delta = step.after.ops - step.before.ops;

  if (step.outcome !== 'changed') {
    return (
      <p className="why-quiet">
        This run recorded no decisions. Passes explain themselves through the trace log, and this one
        had nothing to report for this graph.
      </p>
    );
  }

  return (
    <p className="why-quiet">
      This pass rewrote the {levelLabel(step.level)} without recording a decision — {step.before.ops}{' '}
      nodes in, {step.after.ops} out{delta !== 0 && ` (${delta > 0 ? '+' : ''}${delta})`}. What it did is
      in the IR beside this panel; why it did it never reached the trace.
    </p>
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
          The link survives the boundary: as each op is lowered, the block its rule produces is stamped
          with that op’s identity, so the graph node and the loop nest on the right are matched exactly
          rather than guessed from a name. That matters because names do not survive. <code>dot</code>{' '}
          lowers to blocks called <code>matmul_init_0</code> and <code>matmul_1</code>, and no op in the
          graph is called <code>matmul</code>; a fused group lowers to several blocks, all belonging to
          the one fusion node. What does not survive is the direction — past this point the loop nest is
          the program, and no pass can ask what op it used to be beyond the stamp it is carrying.
        </p>
      </article>
    );
  }

  if (step.kind === 'primitive') {
    return (
      <article className="why-card pass">
        <header>
          <span className="tag">primitive</span>
          <span className="subject">{step.pass}</span>
          <span className="expansion">on {step.parent ?? 'the loop nest'}</span>
        </header>
        <p className="decision">
          One schedule primitive, replayed on the loop nest as it stood before the scheduling pass ran.
          The pass applies the whole sequence at once; this row is what that one call did on its own.
        </p>
        <p className="reason">
          {step.before.ops} IR nodes in, {step.after.ops} out
          {delta !== 0 && ` (${delta > 0 ? '+' : ''}${delta})`} — a primitive changes how the work is
          arranged, never what it computes.
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
        <span className="expansion">{passLabel(step.pass)}</span>
      </header>
      {note && <p className="decision">{note.summary}</p>}
      <p className="reason">
        ran in {phaseLabel(step.phase)} · {step.before.ops} ops in, {step.after.ops} out
        {delta !== 0 && ` (${delta > 0 ? '+' : ''}${delta})`} · {step.durationMs.toFixed(1)}ms
      </p>
      <TrafficNote step={step} />
      {note && (
        <a className="chapter" href={chapterUrl(note)} target="_blank" rel="noreferrer">
          read {note.chapterTitle} →
        </a>
      )}
    </article>
  );
}

function TrafficNote({ step }: { step: CompileStep }) {
  const before = step.before.bytes;
  const after = step.after.bytes;
  if (before === 0 || after === 0 || before === after) return null;

  const ratio = before / after;
  const beforeIntensity = step.before.flops / before;
  const afterIntensity = step.after.flops / after;

  return (
    <p className="reason">
      the ops that survive now read and write {formatMetric('bytes', after)} instead of{' '}
      {formatMetric('bytes', before)}
      {ratio > 1 ? ` — ${ratio.toFixed(2)}× less traffic` : ` — ${(1 / ratio).toFixed(2)}× more traffic`}, at{' '}
      {afterIntensity.toFixed(2)} flop per byte against {beforeIntensity.toFixed(2)} before.
    </p>
  );
}

function EventCard({ group }: { group: Group }) {
  const { event, subjects } = group;
  const times = Math.max(subjects.length, 1);

  if (event.type === 'explain') {
    return (
      <article className="why-card explain">
        <header>
          <span className="tag">{String(event.category)}</span>
          <span className="subject">{summarize(subjects)}</span>
          {times > 1 && <span className="repeat">{times}×</span>}
        </header>
        <p className="decision">{String(event.decision)}</p>
        {event.reason ? <p className="reason">because {String(event.reason)}</p> : null}
        <Extras event={event} skip={EXPLAIN_KEYS} />
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
      <header><span className="tag">{event.type.replace('_', ' ')}</span></header>
      <Extras event={event} skip={BASE_KEYS} />
    </article>
  );
}

function summarize(subjects: readonly string[]): string {
  if (subjects.length === 0) return '';
  if (subjects.length <= SUBJECT_LIMIT) return subjects.join(', ');
  return `${subjects.slice(0, SUBJECT_LIMIT).join(', ')} and ${subjects.length - SUBJECT_LIMIT} more`;
}

function Extras({ event, skip }: { event: TraceEventLite; skip: readonly string[] }) {
  const entries = Object.entries(event).filter(([key, value]) =>
    !skip.includes(key) && !isHiddenMetric(key) && value !== null && value !== undefined);
  if (entries.length === 0) return null;

  return (
    <dl className="extras">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{metricLabel(key)}</dt>
          <dd>{metricValue(key, value)}</dd>
        </div>
      ))}
    </dl>
  );
}
