import { describe, expect, it } from 'vitest';
import { BANNER, getExample, getHelp, handleReplCommand, listExamples } from '../../src/cli/help.js';

describe('Tensor Lang CLI help', () => {
  it('shows a useful first-run example', () => {
    expect(BANNER).toContain('x = tensor([[1, 2], [3, 4]])');
    expect(BANNER).toContain('Type help for examples');
  });

  it('provides topic help', () => {
    expect(getHelp('tensor')).toContain('mean(x, axis=0)');
    expect(getHelp('model')).toContain('forward x');
    expect(getHelp('compile')).toContain('trace(compiled)');
  });

  it('lists and prints examples without executing them', () => {
    expect(listExamples()).toContain('example tensor');
    expect(getExample('custom')).toContain('model MLP');
    expect(handleReplCommand('example compile')).toContain('compile(model, input=x)');
    expect(handleReplCommand('x = tensor(2)')).toBeNull();
  });
});
