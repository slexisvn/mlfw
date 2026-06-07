import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      thresholds: {
        lines: 83,
        functions: 85,
        branches: 65,
        statements: 80,
      },
    },
  },
});
