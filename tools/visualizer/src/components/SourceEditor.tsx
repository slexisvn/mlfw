import { useEffect, useRef } from 'react';
import { EDITOR_THEME, editor, installFrameworkGlobals, KeyCode, KeyMod, setupMonaco } from '../monaco/setup.js';
import { actions, getState, useStore } from '../store.js';

export function SourceEditor() {
  const host = useRef<HTMLDivElement>(null);
  const instance = useRef<editor.IStandaloneCodeEditor | null>(null);
  const globals = useStore(s => s.globals);
  const source = useStore(s => s.source);
  const exampleId = useStore(s => s.exampleId);

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

    return () => {
      subscription.dispose();
      created.dispose();
      instance.current = null;
    };
  }, []);

  useEffect(() => {
    if (globals.length > 0) installFrameworkGlobals(globals);
  }, [globals]);

  useEffect(() => {
    const editor = instance.current;
    if (editor && exampleId && editor.getValue() !== source) editor.setValue(source);
  }, [exampleId, source]);

  return <div className="editor" ref={host} />;
}
