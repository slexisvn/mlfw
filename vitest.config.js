import { defineConfig } from 'vitest/config';

const HARDWARE = ['tests/backend/cuda/**', 'tests/backend/webgpu/**', 'tests/linalg/cuda-linalg.test.js'];

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          include: ['tests/**/*.test.js'],
          exclude: ['tests/e2e/**', 'tests/stress/**', ...HARDWARE],
          testTimeout: 5000,
        },
      },
      {
        test: {
          name: 'e2e',
          include: ['tests/e2e/**/*.test.js'],
          testTimeout: 30000,
        },
      },
      {
        test: {
          name: 'cuda',
          include: ['tests/backend/cuda/**/*.test.js', 'tests/linalg/cuda-linalg.test.js'],
          testTimeout: 60000,
        },
      },
      {
        test: {
          name: 'webgpu',
          include: ['tests/backend/webgpu/**/*.test.js'],
          testTimeout: 60000,
        },
      },
      {
        test: {
          name: 'stress',
          include: ['tests/stress/**/*.test.js'],
          testTimeout: 120000,
        },
      },
    ],
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
