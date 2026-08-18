import { defineConfig } from 'vitest/config';

const CUDA = 'tests/**/*.cuda.test.js';
const WEBGPU = 'tests/**/*.webgpu.test.js';
const SCRATCH = 'tests/**/_*.test.js';
const PERF = 'tests/perf/**';
const STRESS = 'tests/stress/**';

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          include: ['tests/**/*.test.js'],
          exclude: ['tests/e2e/**', STRESS, PERF, CUDA, WEBGPU, SCRATCH],
          testTimeout: 5000,
        },
      },
      {
        test: {
          name: 'e2e',
          include: ['tests/e2e/**/*.test.js'],
          exclude: [CUDA, WEBGPU, SCRATCH],
          testTimeout: 60000,
        },
      },
      {
        test: {
          name: 'cuda',
          include: [CUDA],
          exclude: [PERF, SCRATCH],
          testTimeout: 60000,
          fileParallelism: false,
        },
      },
      {
        test: {
          name: 'webgpu',
          include: [WEBGPU],
          exclude: [PERF, SCRATCH],
          testTimeout: 60000,
        },
      },
      {
        test: {
          name: 'stress',
          include: ['tests/stress/**/*.test.js'],
          exclude: [SCRATCH],
          testTimeout: 120000,
        },
      },
      {
        test: {
          name: 'perf',
          include: ['tests/perf/**/*.test.js'],
          exclude: [SCRATCH],
          testTimeout: 120000,
        },
      },
    ],
    coverage: {
      provider: 'v8',
      include: ['src/**'],
      thresholds: {
        lines: 83,
        functions: 85,
        branches: 65,
        statements: 80,
      },
    },
  },
});
