import terminalKit from 'terminal-kit';
import { formatValue } from './format.js';
import { TensorLangRuntime } from './runtime.js';
import { BANNER, handleReplCommand } from './help.js';
import { formatDiagnostic } from './diagnostics.js';

const KEYWORDS = new Set(['model', 'forward', 'return', 'true', 'false', 'null']);
const COMMANDS = ['help', 'help tensor', 'help model', 'help compile', 'examples',
  'example tensor', 'example linear', 'example custom', 'example compile', 'exit', 'quit'];

export async function startRepl({ term = terminalKit.terminal } = {}) {
  const write = text => term(String(text) + '\n');
  const runtime = new TensorLangRuntime({ output: write });
  const history = [];
  let buffer = '';
  let depth = 0;
  let interrupted = false;

  term(BANNER + '\n');

  while (true) {
    term(depth > 0 ? '^K...   ^:' : '^Kmlfw> ^:');
    const controller = term.inputField({
      history,
      autoComplete: input => completeInput(input, runtime),
      autoCompleteHint: true,
      autoCompleteMenu: {
        style: term.brightBlack,
        selectedStyle: term.bgCyan.black,
      },
      cancelable: true,
      autoClosePairs: {
        '(': ')',
        '[': ']',
        '{': '}',
        '"': '"',
        "'": "'",
      },
      tokenRegExp: /#[^\n]*|"(?:\\.|[^"])*"|'(?:\\.|[^'])*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*\b|==|!=|<=|>=|\*\*|[^\s]/g,
      tokenHook,
    });
    const onKey = key => {
      if (key !== 'CTRL_C') return;
      interrupted = true;
      controller.stop();
    };
    const onSigint = () => {
      interrupted = true;
      controller.stop();
    };
    term.on('key', onKey);
    process.once('SIGINT', onSigint);

    let line;
    try {
      line = await controller.promise;
    } finally {
      term.removeListener('key', onKey);
      process.removeListener('SIGINT', onSigint);
    }
    term('\n');
    if (interrupted) {
      shutdownTerminal(term);
      term.brightBlack('^C\n');
      process.exitCode = 130;
      break;
    }
    if (line === undefined || (!buffer && (line.trim() === 'exit' || line.trim() === 'quit'))) {
      term.brightBlack('Bye.\n');
      break;
    }
    if (line.trim()) history.push(line);

    if (!buffer) {
      const commandOutput = handleReplCommand(line);
      if (commandOutput !== null) {
        write(commandOutput);
        continue;
      }
    }

    buffer += line + '\n';
    depth += braceDelta(line);
    if (depth > 0) continue;

    try {
      const value = runtime.execute(buffer);
      const text = formatValue(value);
      if (text) write(text);
    } catch (error) {
      term.red(`${formatDiagnostic(error, buffer)}\n`);
    }
    buffer = '';
    depth = 0;
  }

  shutdownTerminal(term);
  return { runtime, history };
}

export function shutdownTerminal(term) {
  if (typeof term.grabInput === 'function') term.grabInput(false);
  if (typeof term.hideCursor === 'function') term.hideCursor(false);
  if (typeof term.styleReset === 'function') term.styleReset();
}

export function completeInput(input, runtime) {
  const match = input.match(/[A-Za-z_][A-Za-z0-9_]*$/);
  if (!match) return input;

  const prefix = input.slice(0, match.index);
  const word = match[0];
  const candidates = [...new Set([...COMMANDS, ...KEYWORDS, ...runtime.getCompletionNames()])]
    .filter(name => name.startsWith(word))
    .sort();

  if (candidates.length === 0) return input;
  if (candidates.length === 1) return prefix + candidates[0];
  const common = commonPrefix(candidates);
  if (common.length > word.length) return prefix + common;
  candidates.prefix = prefix;
  return candidates;
}

function commonPrefix(values) {
  let prefix = values[0] || '';
  for (const value of values.slice(1)) {
    while (!value.startsWith(prefix)) prefix = prefix.slice(0, -1);
  }
  return prefix;
}

export function tokenHook(token, _isEnd, _previous, term) {
  if (token.startsWith('#')) return term.brightBlack;
  if ((token.startsWith('"') && token.endsWith('"')) || (token.startsWith("'") && token.endsWith("'"))) return term.green;
  if (/^\d+(?:\.\d+)?$/.test(token)) return term.yellow;
  if (KEYWORDS.has(token)) return term.brightMagenta;
  if (/^(?:==|!=|<=|>=|\*\*|[+\-*/@=<>])$/.test(token)) return term.brightCyan;
  if (/^[A-Z]/.test(token)) return term.brightBlue;
  return term;
}

function braceDelta(line) {
  let depth = 0;
  let quote = null;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quote) {
      if (ch === '\\') i++;
      else if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '{' || ch === '(' || ch === '[') depth++;
    else if (ch === '}' || ch === ')' || ch === ']') depth--;
  }
  return depth;
}
