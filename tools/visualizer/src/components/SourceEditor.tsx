import { useEffect, useRef } from 'react';
import { EDITOR_THEME, editor, installFrameworkGlobals, KeyCode, KeyMod, Range, setupMonaco } from '../monaco/setup.js';
import { actions, getState, sourceLines, useStore } from '../store.js';

const FOCUS_CLEAR_MS = 1800;

export function SourceEditor() {
  const host = useRef<HTMLDivElement>(null);
  const instance = useRef<editor.IStandaloneCodeEditor | null>(null);
  const traced = useRef<editor.IEditorDecorationsCollection | null>(null);
  const focused = useRef<editor.IEditorDecorationsCollection | null>(null);
  const globals = useStore(s => s.globals);
  const source = useStore(s => s.source);
  const exampleId = useStore(s => s.exampleId);
  const lines = useStore(sourceLines);
  const focusLine = useStore(s => s.focusLine);

  useEffect(() => {
    const node = host.current;
    if (!node) return;

    setupMonaco();
    const created = editor.create(node, {
      value: getState().source,
      language: 'javascript',
      theme: EDITOR_THEME,
      automaticLayout: true,
      minimap: { enabled: false },
      fontSize: 13,
      lineNumbersMinChars: 3,
      scrollBeyondLastLine: false,
      tabSize: 2,
      padding: { top: 12 },
      scrollbar: { verticalScrollbarSize: 7, horizontalScrollbarSize: 7, useShadows: false },
    });

    created.addCommand(KeyMod.CtrlCmd | KeyCode.Enter, () => { void actions.run(); });
    const subscription = created.onDidChangeModelContent(() => actions.setSource(created.getValue()));
    instance.current = created;
    traced.current = created.createDecorationsCollection();
    focused.current = created.createDecorationsCollection();

    return () => {
      subscription.dispose();
      created.dispose();
      instance.current = null;
      traced.current = null;
      focused.current = null;
    };
  }, []);

  useEffect(() => {
    if (globals.length > 0) installFrameworkGlobals(globals);
  }, [globals]);

  useEffect(() => {
    const editorInstance = instance.current;
    if (editorInstance && exampleId && editorInstance.getValue() !== source) editorInstance.setValue(source);
  }, [exampleId, source]);

  useEffect(() => {
    const collection = traced.current;
    if (!collection) return;
    collection.set(lines.map(line => ({
      range: new Range(line, 1, line, 1),
      options: {
        isWholeLine: true,
        className: 'traced-line',
        linesDecorationsClassName: 'traced-gutter',
        hoverMessage: { value: 'this line produced ops in the graph' },
      },
    })));
  }, [lines]);

  useEffect(() => {
    const editorInstance = instance.current;
    const collection = focused.current;
    if (!editorInstance || !collection || focusLine === null) return;

    editorInstance.revealLineInCenterIfOutsideViewport(focusLine);
    collection.set([{
      range: new Range(focusLine, 1, focusLine, 1),
      options: { isWholeLine: true, className: 'focused-line' },
    }]);

    const timer = setTimeout(() => {
      collection.clear();
      actions.focusSource(null);
    }, FOCUS_CLEAR_MS);

    return () => clearTimeout(timer);
  }, [focusLine]);

  return <div className="editor" ref={host} />;
}
