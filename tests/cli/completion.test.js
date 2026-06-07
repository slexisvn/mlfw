import { describe, expect, it } from 'vitest';
import { completeInput, shutdownTerminal, tokenHook } from '../../src/cli/repl.js';
import { TensorLangRuntime } from '../../src/cli/runtime.js';

describe('Tensor Lang completion', () => {
  it('completes builtins and preserves the expression prefix', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(completeInput('rel', runtime)).toBe('relu');
    expect(completeInput('x = rand', runtime)).toBe('x = randn');
    expect(completeInput('mod', runtime)).toBe('model');
    expect(completeInput('ret', runtime)).toBe('return');
  });

  it('includes names created during the session', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    runtime.execute('weights = randn([2, 2])');
    expect(completeInput('wei', runtime)).toBe('weights');
  });

  it('completes user-defined functions and model names', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    runtime.execute('fn my_normalize(x): return x / sum(x)');
    expect(completeInput('my_n', runtime)).toBe('my_normalize');

    runtime.execute(`model MyNet:
  forward x:
    return relu(x)`);
    expect(completeInput('MyN', runtime)).toBe('MyNet');
  });

  it('completes properties of custom model instances', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    runtime.execute(`model MLP(h):
  fc1 = Linear(2, h)
  fc2 = Linear(h, 1)
  forward x:
    return fc2(relu(fc1(x)))`);
    runtime.execute('net = MLP(4)');

    expect(completeInput('net.f', runtime)).toBe('net.fc');
    const props = completeInput('net.fc', runtime);
    expect(Array.isArray(props)).toBe(true);
    expect(props).toContain('fc1');
    expect(props).toContain('fc2');
  });

  it('completes properties of builtin modules', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    runtime.execute('layer = Linear(4, 2)');
    expect(completeInput('layer.w', runtime)).toBe('layer.weight');
  });

  it('suggests names from the current buffer before execution', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const buffer = `model MLP(h):
  fc1 = Linear(2, h)
  fc2 = Linear(h, 1)
  forward x:
`;
    // fc1, fc2 not yet executed, but extractable from buffer
    const result = completeInput('fc', runtime, buffer);
    expect(Array.isArray(result)).toBe(true);
    expect(result).toContain('fc1');
    expect(result).toContain('fc2');

    // model name from buffer
    expect(completeInput('ML', runtime, buffer)).toBe('MLP');

    // forward param from buffer
    expect(completeInput('x = f', runtime, buffer)).toContain('fc1');
  });

  it('returns alternatives for ambiguous input', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = completeInput('help ', runtime);
    expect(result).toBe('help ');

    const alternatives = completeInput('ex', runtime);
    expect(Array.isArray(alternatives)).toBe(true);
    expect(alternatives).toContain('examples');
    expect(alternatives).toContain('exit');
  });

  it('assigns syntax styles through terminal-kit token hooks', () => {
    const term = {
      brightBlack: 'dim',
      green: 'string',
      magenta: 'number',
      brightBlue: 'keyword',
      yellow: 'builtin',
      cyan: 'type',
    };
    expect(tokenHook('model', false, [], term)).toBe('keyword');
    expect(tokenHook('42', false, [], term)).toBe('number');
    expect(tokenHook('@', false, [], term)).toBe('dim');
    expect(tokenHook('Linear', false, [], term)).toBe('type');
    expect(tokenHook('relu', false, [], term)).toBe('builtin');
    expect(tokenHook('tensor', false, [], term)).toBe('builtin');
    expect(tokenHook('print', false, [], term)).toBe('builtin');
  });

  it('restores terminal state before exiting', () => {
    const calls = [];
    const term = {
      grabInput: value => calls.push(['grabInput', value]),
      hideCursor: value => calls.push(['hideCursor', value]),
      styleReset: () => calls.push(['styleReset']),
    };
    shutdownTerminal(term);
    expect(calls).toEqual([['grabInput', false], ['hideCursor', false], ['styleReset']]);
  });

});
