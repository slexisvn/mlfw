import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { transform } from 'esbuild';

const here = dirname(fileURLToPath(import.meta.url));
const buildScript = resolve(here, '../../scripts/build.js');

const NODE_SNIPPET =
  `(() => { class ForNode { constructor(){ this.type = this.constructor.name.replace(/^_+/, ''); } } return new ForNode().type; })()`;

describe('minified bundle keeps class names for node.type dispatch', () => {
  it('build.js sets keepNames so the minified bundle preserves constructor.name', () => {
    const src = readFileSync(buildScript, 'utf8');
    expect(src).toMatch(/keepNames:\s*true/);
  });

  it('minify + keepNames preserves constructor.name', async () => {
    const { code } = await transform(NODE_SNIPPET, { minify: true, keepNames: true });
    expect((0, eval)(code)).toBe('ForNode');
  });

  it('minify alone mangles the class name, so keepNames is load-bearing', async () => {
    const { code } = await transform(NODE_SNIPPET, { minify: true });
    expect((0, eval)(code)).not.toBe('ForNode');
  });
});
