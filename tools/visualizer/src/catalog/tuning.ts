export const SEARCH_BUDGET = {
  numTrials: 24,
  populationSize: 12,
  numGenerations: 4,
  timeBudgetMs: 4000,
  useTuningDB: false,
} as const;

const SKETCH_NOTES: Record<string, string> = {
  fused: 'folds this block into the one that consumes it so both share a single loop nest',
  rfactor: 'cuts the reduction into parallel partial sums, then combines them',
  elementwise_cpu: 'runs the outer loop across cores and the innermost loop as vector lanes',
  elementwise_gpu: 'flattens every loop into one and hands each element to its own thread',
  reduction_cpu: 'spreads the spatial loop across cores and keeps the reduction serial inside each one',
  reduction_gpu: 'gives each output element a thread and lets that thread walk the reduction',
  matmul_register_block_gpu: 'stages tiles through shared memory and keeps a block of the output in registers',
};

const KNOB_NOTES: Record<string, string> = {
  block_size: 'threads per GPU block',
  vector_width: 'lanes per vector instruction',
  rf_factor: 'pieces the reduction is cut into',
};

export function sketchNote(name: string): string | null {
  return SKETCH_NOTES[name] ?? null;
}

export function knobNote(name: string): string | null {
  return KNOB_NOTES[name] ?? null;
}
