import { readdirSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

const ARTIFACTS = ['dist/index.node.js', 'dist/internals.node.js'];
const SOURCES = ['src', 'docs/tools/internals-entry.ts'];

function newestUnder(path, cutoff) {
  const stat = statSync(path);
  if (!stat.isDirectory()) {
    return path.endsWith('.ts') && stat.mtimeMs > cutoff ? path : null;
  }
  for (const entry of readdirSync(path, { withFileTypes: true })) {
    const hit = newestUnder(join(path, entry.name), cutoff);
    if (hit) return hit;
  }
  return null;
}

export function requireFreshBuild() {
  for (const artifact of ARTIFACTS) {
    let built;
    try {
      built = statSync(join(root, artifact)).mtimeMs;
    } catch {
      throw new Error(`docs: no build found at ${artifact}\n  run: npm run build`);
    }
    for (const source of SOURCES) {
      const stale = newestUnder(join(root, source), built);
      if (!stale) continue;
      throw new Error(
        `docs: ${artifact} is older than ${relative(root, stale)}\n`
        + '  The labs read the built package, so they would report a compiler that no\n'
        + '  longer exists — and the numbers that go wrong when it does go wrong quietly.\n'
        + '  run: npm run build',
      );
    }
  }
}

requireFreshBuild();
