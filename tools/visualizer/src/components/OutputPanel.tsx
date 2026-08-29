import { useEffect, useRef, useState } from 'react';
import { EDITOR_THEME, editor, setupMonaco } from '../monaco/setup.js';
import { plural } from '../catalog/naming.js';
import { KERNEL_METRICS } from '../catalog/diagnostics.js';
import type { Kernel, KernelReport } from '../protocol.js';

const GRAMMAR: Record<string, string> = {
  javascript: 'javascript',
  cpp: 'cpp',
  wgsl: 'wgsl',
  wat: 'scheme',
};

const COPY_RESET_MS = 1600;

export function OutputPanel({ kernels, reports }: { kernels: readonly Kernel[]; reports: readonly KernelReport[] }) {
  const [active, setActive] = useState(0);
  const [copied, setCopied] = useState(false);

  if (kernels.length === 0) {
    return <div className="pane-empty">No kernel yet — run a compile and the generated source lands here.</div>;
  }

  const kernel = kernels[Math.min(active, kernels.length - 1)];

  const copy = async (): Promise<void> => {
    await navigator.clipboard.writeText(kernel.source);
    setCopied(true);
    setTimeout(() => setCopied(false), COPY_RESET_MS);
  };

  return (
    <div className="output">
      <div className="kernel-tabs">
        {kernels.length > 1
          ? kernels.map((k, i) => (
            <button key={k.name} className={i === active ? 'active' : ''} onClick={() => setActive(i)}>
              {k.name}
            </button>
          ))
          : <span className="kernel-name">{kernel.name}</span>}
        <span className="kernel-lang">{kernel.language}</span>
        <button className="copy" onClick={() => { void copy(); }}>
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <Report report={reports.find(entry => entry.name === kernel.name) ?? null} kernel={kernel} />
      <KernelSource kernel={kernel} />
    </div>
  );
}

function Report({ report, kernel }: { report: KernelReport | null; kernel: Kernel }) {
  if (!report) return null;

  const notes: string[] = [];
  if (report.extent1Loops > 0) notes.push(`${plural(report.extent1Loops, 'loop')} that run once`);
  if (report.zeroInits > 0) notes.push(plural(report.zeroInits, 'zero fill'));
  if (report.arithmeticNoise.length > 0) notes.push(`arithmetic left in: ${report.arithmeticNoise.join(', ')}`);

  return (
    <div className={report.blownUp || report.issues.length > 0 || kernel.diagnosis ? 'kernel-report alarming' : 'kernel-report'}>
      <dl className="extras kernel-metrics">
        {KERNEL_METRICS.map(metric => (
          <div key={metric.key} className={metric.notable(report) ? 'notable' : ''} title={metric.meaning}>
            <dt>{metric.label}</dt>
            <dd>{report[metric.key].toLocaleString()}</dd>
          </div>
        ))}
      </dl>
      {kernel.diagnosis && (
        <p className="kernel-alarm">
          The compiler refused to launch this kernel in parallel: {kernel.diagnosis.reason}
          {kernel.diagnosis.buffers.length > 0 && ` — ${kernel.diagnosis.buffers.join(', ')}`}.
          It runs correctly and serially, which is the compiler choosing to be right rather than fast.
        </p>
      )}
      {report.blownUp && (
        <p className="kernel-alarm">
          One line is {report.longestLine.toLocaleString()} characters long. That is the shape of an
          expression duplicated by codegen rather than named once — the failure mode that has produced
          multi-megabyte kernels here before.
        </p>
      )}
      {report.issues.length > 0 && (
        <ul className="invariant-list">
          {report.issues.map((issue, i) => <li key={i}>{issue.kind}: {issue.detail}</li>)}
        </ul>
      )}
      {notes.length > 0 && <p className="kernel-notes">{notes.join(' · ')}</p>}
    </div>
  );
}

function KernelSource({ kernel }: { kernel: Kernel }) {
  const host = useRef<HTMLDivElement>(null);
  const instance = useRef<editor.IStandaloneCodeEditor | null>(null);

  useEffect(() => {
    const node = host.current;
    if (!node) return;

    setupMonaco();
    const created = editor.create(node, {
      value: '',
      theme: EDITOR_THEME,
      automaticLayout: true,
      readOnly: true,
      minimap: { enabled: false },
      fontSize: 12,
      scrollBeyondLastLine: false,
      scrollbar: { verticalScrollbarSize: 7, horizontalScrollbarSize: 7, useShadows: false },
    });
    instance.current = created;

    return () => { created.dispose(); instance.current = null; };
  }, []);

  useEffect(() => {
    const editorInstance = instance.current;
    if (!editorInstance) return;
    const model = editorInstance.getModel();
    if (model) editor.setModelLanguage(model, GRAMMAR[kernel.language] ?? 'plaintext');
    if (editorInstance.getValue() !== kernel.source) editorInstance.setValue(kernel.source);
  }, [kernel]);

  return <div className="output-host" ref={host} />;
}
