import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

export default defineConfig({
  base: process.env.VITE_BASE || '/',
  plugins: [react()],
  resolve: {
    alias: [
      { find: /^mlfw\//, replacement: `${repoRoot}/src/` },
      { find: /^mlfw-dist\//, replacement: `${repoRoot}/dist/` },
      { find: /^mlfw-tests\//, replacement: `${repoRoot}/tests/` },
    ],
  },
  server: {
    fs: { allow: [repoRoot] },
  },
  optimizeDeps: { exclude: ['monaco-editor'] },
  worker: {
    format: 'es',
    rolldownOptions: { output: { keepNames: true } },
  },
  build: {
    target: 'es2022',
    rolldownOptions: { output: { keepNames: true } },
  },
});
