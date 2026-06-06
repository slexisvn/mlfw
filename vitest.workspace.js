import { defineWorkspace } from 'vitest/config';

export default defineWorkspace([
  {
    test: {
      name: 'unit',
      include: ['tests/**/*.test.js'],
      exclude: ['tests/e2e/**'],
      setupFiles: ['tests/_utils/setup.js'],
      testTimeout: 5000,
    },
  },
  {
    test: {
      name: 'e2e',
      include: ['tests/e2e/**/*.test.js'],
      setupFiles: ['tests/_utils/setup.js'],
      testTimeout: 30000,
    },
  },
]);
