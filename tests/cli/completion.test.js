import { describe, expect, it } from 'vitest';
import { completeInput, shutdownTerminal, tokenHook } from '../../src/cli/repl.js';
import { TensorLangRuntime } from '../../src/cli/runtime.js';

describe('Tensor Lang completion', () => {
  it('completes builtins and preserves the expression prefix', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(completeInput('rel', runtime)).toBe('relu');
    expect(completeInput('x = ran', runtime)).toBe('x = randn');
    expect(completeInput('mod', runtime)).toBe('model');
    expect(completeInput('ret', runtime)).toBe('return');
  });

  it('includes names created during the session', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    runtime.execute('weights = randn([2, 2])');
    expect(completeInput('wei', runtime)).toBe('weights');
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
      brightBlack: 'comment',
      green: 'string',
      yellow: 'number',
      brightMagenta: 'keyword',
      brightCyan: 'operator',
      brightBlue: 'type',
    };
    expect(tokenHook('model', false, [], term)).toBe('keyword');
    expect(tokenHook('42', false, [], term)).toBe('number');
    expect(tokenHook('@', false, [], term)).toBe('operator');
    expect(tokenHook('Linear', false, [], term)).toBe('type');
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
