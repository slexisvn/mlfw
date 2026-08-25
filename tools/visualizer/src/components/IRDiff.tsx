import { useCallback, useEffect, useRef, useState } from 'react';
import { EDITOR_THEME, editor, IR_LANGUAGE, setupMonaco } from '../monaco/setup.js';
import { useElementSize } from './use_element_size.js';
import type { CompileStep } from '../protocol.js';

const SIDE_BY_SIDE_MIN_PX = 820;
const FALLBACK_SIZE = { width: SIDE_BY_SIDE_MIN_PX, height: 400 };

export function IRDiff({ step }: { step: CompileStep }) {
  const diff = useRef<editor.IStandaloneDiffEditor | null>(null);
  const models = useRef<{ original: editor.ITextModel; modified: editor.ITextModel } | null>(null);
  const [preferSideBySide, setPreferSideBySide] = useState(true);
  const { size, ref: measure } = useElementSize<HTMLDivElement>(FALLBACK_SIZE);
  const identical = step.before.text === step.after.text;
  const roomy = size.width >= SIDE_BY_SIDE_MIN_PX;
  const sideBySide = preferSideBySide && roomy && !identical;

  const host = useCallback((node: HTMLDivElement | null) => {
    measure(node);
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
  }, [measure]);

  useEffect(() => () => {
    diff.current?.dispose();
    models.current?.original.dispose();
    models.current?.modified.dispose();
    diff.current = null;
    models.current = null;
  }, []);

  useEffect(() => {
    const pair = models.current;
    if (!pair) return;
    if (pair.original.getValue() !== step.before.text) pair.original.setValue(step.before.text);
    if (pair.modified.getValue() !== step.after.text) pair.modified.setValue(step.after.text);
    diff.current?.revealFirstDiff?.();
  }, [step]);

  useEffect(() => {
    diff.current?.updateOptions({ renderSideBySide: sideBySide });
  }, [sideBySide]);

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
          <button
            disabled={!roomy}
            title={roomy ? undefined : 'the pane is too narrow to show two columns'}
            onClick={() => setPreferSideBySide(v => !v)}
          >
            {sideBySide ? 'inline' : 'side by side'}
          </button>
        )}
      </div>
      <div className="irdiff-host" ref={host} />
    </div>
  );
}
