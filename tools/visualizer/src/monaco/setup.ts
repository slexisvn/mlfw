import { editor, languages, KeyCode, KeyMod, typescript } from 'monaco-editor';
import editorWorker from 'monaco-editor/editor/editor.worker.js?worker';
import tsWorker from 'monaco-editor/languages/features/typescript/ts.worker.js?worker';
import mlfwTypes from 'mlfw-dist/index.d.ts?raw';

export { editor, languages, KeyCode, KeyMod };

const { javascriptDefaults, ModuleKind, ModuleResolutionKind, ScriptTarget } = typescript;

export const IR_LANGUAGE = 'mlfw-ir';
export const EDITOR_THEME = 'mlfw-dark';

const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const MLFW_TYPES_PATH = 'file:///node_modules/mlfw/index.d.ts';
const GLOBALS_PATH = 'file:///mlfw-globals.d.ts';

let installed = false;

export function setupMonaco(): void {
  if (installed) return;
  installed = true;

  self.MonacoEnvironment = {
    getWorker(_workerId: string, label: string) {
      if (label === 'typescript' || label === 'javascript') return new tsWorker();
      return new editorWorker();
    },
  };

  javascriptDefaults.setCompilerOptions({
    target: ScriptTarget.ESNext,
    module: ModuleKind.ESNext,
    moduleResolution: ModuleResolutionKind.NodeJs,
    allowNonTsExtensions: true,
    allowJs: true,
    checkJs: false,
    lib: ['es2022'],
  });
  javascriptDefaults.setDiagnosticsOptions({ noSemanticValidation: true, noSyntaxValidation: false });
  javascriptDefaults.addExtraLib(mlfwTypes, MLFW_TYPES_PATH);

  registerIRLanguage();
  registerTheme();
}

export function installFrameworkGlobals(names: readonly string[]): void {
  const declarations = names
    .filter(name => IDENTIFIER.test(name))
    .map(name => `  const ${name}: typeof M.${name};`)
    .join('\n');

  javascriptDefaults.addExtraLib(
    [
      `import * as M from 'mlfw';`,
      'declare global {',
      declarations,
      '  function run(model: unknown, inputs: unknown[]): void;',
      '}',
      'export {};',
    ].join('\n'),
    GLOBALS_PATH,
  );
}

function registerIRLanguage(): void {
  languages.register({ id: IR_LANGUAGE });

  languages.setMonarchTokensProvider(IR_LANGUAGE, {
    defaultToken: '',
    tokenizer: {
      root: [
        [/\b(module|func|region)\b/, 'keyword'],
        [/@[A-Za-z_][\w.$]*/, 'type.identifier'],
        [/%[\w.]+/, 'variable'],
        [/\^bb\d*/, 'keyword.control'],
        [/<[^>]*>/, 'type'],
        [/"[^"]*"/, 'string'],
        [/\b(inf|-inf|nan|true|false)\b/, 'number'],
        [/\b\d+(\.\d+)?([eE][-+]?\d+)?\b/, 'number'],
        [/[A-Za-z_][\w.]*(?=\()/, 'entity.name.function'],
        [/[{}()[\]]/, 'delimiter'],
        [/[=:,]/, 'delimiter'],
      ],
    },
  });
}

function registerTheme(): void {
  editor.defineTheme(EDITOR_THEME, {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'keyword', foreground: 'c792ea' },
      { token: 'variable', foreground: '82aaff' },
      { token: 'type', foreground: '7fdbca' },
      { token: 'type.identifier', foreground: 'ffcb6b' },
      { token: 'entity.name.function', foreground: 'f78c6c' },
      { token: 'number', foreground: 'f78c6c' },
      { token: 'string', foreground: 'c3e88d' },
    ],
    colors: {
      'editor.background': '#0f1117',
      'editorGutter.background': '#0f1117',
      'scrollbar.shadow': '#00000000',
      'scrollbarSlider.background': '#2b3348aa',
      'scrollbarSlider.hoverBackground': '#3d4967cc',
      'scrollbarSlider.activeBackground': '#82aaffaa',
    },
  });
}
