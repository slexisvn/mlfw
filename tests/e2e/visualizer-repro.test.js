import { describe, it, expect, afterAll } from 'vitest';
import { execFileSync } from 'node:child_process';
import { writeFileSync, rmSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { DEFAULT_OPTIONS } from '../../tools/visualizer/src/protocol.js';
import { reproScript } from '../../tools/visualizer/src/repro.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const SCRIPT = join(ROOT, '.visualizer-repro.test.mjs');

const SOURCE = `const model = new Sequential(
  new Linear(8, 16),
  new ReLU(),
  new Linear(16, 4),
);

const x = randn([1, 8]);

run(model, [x]);
`;

function runRepro(options) {
  writeFileSync(SCRIPT, reproScript(SOURCE, options), 'utf8');
  return execFileSync(process.execPath, [SCRIPT], { cwd: ROOT, encoding: 'utf8', timeout: 60000 });
}

afterAll(() => {
  if (existsSync(SCRIPT)) rmSync(SCRIPT);
});

describe('the repro the visualizer exports', () => {
  it('runs under plain node and reproduces the compile', () => {
    const output = runRepro(DEFAULT_OPTIONS);

    expect(output).toContain('compiled and eager agree');
    expect(output).toContain('kernels');
  });

  it('carries the disabled passes across, at every IR level', () => {
    const output = runRepro({ ...DEFAULT_OPTIONS, disabledPasses: ['dce', 'SimplifyPass'] });

    expect(output).toContain('[skipped] dce graph-module');
    expect(output).toContain('[skipped] SimplifyPass tir');
    expect(output).toContain('compiled and eager agree');
  });
});
