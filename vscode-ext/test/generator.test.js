import { describe, it, expect } from 'vitest';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { generate } from '../scripts/generate.js';
import { readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, '../..');

describe('generate()', () => {
  it('produces deterministic outputs from canonical sources', () => {
    const tmp = mkdtempSync(join(tmpdir(), 'tera-gen-'));
    const outputs = {
      grammar: join(tmp, 'grammar.json'),
      languageData: join(tmp, 'language-data.json'),
      snippets: join(tmp, 'snippets.json'),
    };
    const sources = {
      parser: join(REPO_ROOT, 'src/cli/parser.js'),
      tokenizer: join(REPO_ROOT, 'src/cli/tokenizer.js'),
      builtins: join(REPO_ROOT, 'src/cli/builtins.js'),
      builtinDocs: join(REPO_ROOT, 'vscode-ext/data/builtin-docs.md'),
    };
    const first = generate(sources, outputs);
    const second = generate(sources, outputs);
    expect(second.builtins.length).toBe(first.builtins.length);
    expect(first.keywords).toContain('model');
    expect(first.keywords).toContain('forward');
    expect(first.builtins.find(b => b.name === 'Linear')).toBeTruthy();
    expect(first.builtins.find(b => b.name === 'Adam')).toBeTruthy();
    const data = JSON.parse(readFileSync(outputs.languageData, 'utf8'));
    expect(data.builtins.find(b => b.name === 'Linear').signature.display).toMatch(/Linear\(/);
    expect(data.builtins.find(b => b.name === 'Linear').description).toBeTruthy();
    expect(data.builtins.find(b => b.name === 'cpu').description).toBeTruthy();
  });
});
