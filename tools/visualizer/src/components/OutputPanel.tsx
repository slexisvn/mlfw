import { useEffect, useRef, useState } from 'react';
import { EDITOR_THEME, editor, setupMonaco } from '../monaco/setup.js';
import type { Kernel } from '../protocol.js';

const GRAMMAR: Record<string, string> = {
  javascript: 'javascript',
  cpp: 'cpp',
  wgsl: 'wgsl',
  wat: 'scheme',
};

export function OutputPanel({ kernels }: { kernels: readonly Kernel[] }) {
  const [active, setActive] = useState(0);

  if (kernels.length === 0) {
    return <div className="pane-empty">No kernel yet — run a compile and the generated source lands here.</div>;
  }

  const kernel = kernels[Math.min(active, kernels.length - 1)];

  return (
    <div className="output">
      <div className="kernel-tabs">
        {kernels.map((k, i) => (
          <button key={k.name} className={i === active ? 'active' : ''} onClick={() => setActive(i)}>
            {k.name}
          </button>
        ))}
        <span className="kernel-lang">{kernel.language}</span>
      </div>
      <KernelSource kernel={kernel} />
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
