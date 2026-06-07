import fs from 'node:fs';
import { parse } from './parser.js';
import { TensorLangRuntime } from './runtime.js';
import { formatValue } from './format.js';
import { formatDiagnostic } from './diagnostics.js';
import { startRepl } from './repl.js';

export const CLI_USAGE = `Usage:
  mlfw                 Start the Tensor Lang REPL
  mlfw repl            Start the Tensor Lang REPL
  mlfw run <file>      Execute a Tensor Lang file
  mlfw check <file>    Parse a Tensor Lang file without executing it
  mlfw <file>          Execute a Tensor Lang file`;

export async function runCli(args, {
  stdout = console.log,
  stderr = console.error,
  readFile = file => fs.readFileSync(file, 'utf8'),
  repl = startRepl,
  stdinIsTTY = process.stdin.isTTY,
  readStdin = () => readStream(process.stdin),
} = {}) {
  const [command, operand, ...extra] = args;
  if (!command || command === 'repl') {
    if (operand || extra.length) return fail(stderr, 'repl does not accept arguments');
    if (!stdinIsTTY) {
      return await executeSource(await readStdin(), { stdout, stderr, filename: '<stdin>', stripExit: true });
    }
    await repl();
    return 0;
  }
  if (command === '--help' || command === '-h' || command === 'help') {
    stdout(CLI_USAGE);
    return 0;
  }

  const mode = command === 'run' || command === 'check' ? command : 'run';
  const file = mode === command ? operand : command;
  if (!file || extra.length || (mode !== command && operand)) {
    return fail(stderr, `Invalid arguments.\n\n${CLI_USAGE}`);
  }

  let source;
  try {
    source = readFile(file);
    if (mode === 'check') {
      parse(source);
      stdout(`${file}: OK`);
      return 0;
    }
    return await executeSource(source, { stdout, stderr, filename: file });
  } catch (error) {
    stderr(source === undefined ? `${error.name || 'Error'}: ${error.message}` : formatDiagnostic(error, source, file));
    return 1;
  }
}

export async function executeSource(source, {
  stdout = console.log,
  stderr = console.error,
  filename = null,
  stripExit = false,
} = {}) {
  if (stripExit) source = source.replace(/(?:^|\n)\s*(?:exit|quit)\s*;?\s*$/u, '');
  try {
    const result = await new TensorLangRuntime({ output: stdout }).execute(source);
    const text = formatValue(result);
    if (text) stdout(text);
    return 0;
  } catch (error) {
    stderr(formatDiagnostic(error, source, filename));
    return 1;
  }
}

function fail(stderr, message) {
  stderr(message);
  return 1;
}

async function readStream(stream) {
  let source = '';
  for await (const chunk of stream) source += chunk;
  return source;
}
