import { useEffect, useRef, useState } from 'react';
import { EDITOR_THEME, editor, IR_LANGUAGE, setupMonaco } from '../monaco/setup.js';
import type { CompileStep } from '../protocol.js';

export function IRDiff({ step }: { step: CompileStep }) {
  const host = useRef<HTMLDivElement>(null);
  const diff = useRef<editor.IStandaloneDiffEditor | null>(null);
  const models = useRef<{ original: editor.ITextModel; modified: editor.ITextModel } | null>(null);
  const [sideBySide, setSideBySide] = useState(true);
  const identical = step.before.text === step.after.text;

  useEffect(() => {
    const node = host.current;
    if (!node) return;

    setupMonaco();
    const original = editor.createModel('', IR_LANGUAGE);
    const modified = editor.createModel('', IR_LANGUAGE);
    const created = editor.createDiffEditor(node, {
      theme: EDITOR_THEME,
      automaticLayout: true,
      readOnly: true,
      originalEditable: false,
      renderSideBySide: true,
      minimap: { enabled: false },
      fontSize: 12,
      scrollBeyondLastLine: false,
      renderOverviewRuler: false,
      scrollbar: { verticalScrollbarSize: 7, horizontalScrollbarSize: 7, useShadows: false },
    });
    created.setModel({ original, modified });

    diff.current = created;
    models.current = { original, modified };

    return () => {
      created.dispose();
      original.dispose();
      modified.dispose();
      diff.current = null;
      models.current = null;
    };
  }, []);

  useEffect(() => {
    const pair = models.current;
    if (!pair) return;
    if (pair.original.getValue() !== step.before.text) pair.original.setValue(step.before.text);
    if (pair.modified.getValue() !== step.after.text) pair.modified.setValue(step.after.text);
    diff.current?.revealFirstDiff?.();
  }, [step]);

  useEffect(() => {
    diff.current?.updateOptions({ renderSideBySide: sideBySide && !identical });
  }, [sideBySide, identical]);

  return (
    <div className="irdiff">
      <div className="irdiff-bar">
        <span className="irdiff-label">
          {step.kind === 'lowering'
            ? <><strong>{step.pass}</strong> — the same program, a different language</>
            : step.kind === 'input'
            ? <>the graph as <strong>traced</strong>, before any pass ran</>
            : identical
              ? <><strong>{step.pass}</strong> left the IR untouched</>
              : <>before <strong>{step.pass}</strong> → after</>}
        </span>
        {!identical && (
          <button onClick={() => setSideBySide(v => !v)}>
            {sideBySide ? 'inline' : 'side by side'}
          </button>
        )}
      </div>
      <div className="irdiff-host" ref={host} />
    </div>
  );
}
