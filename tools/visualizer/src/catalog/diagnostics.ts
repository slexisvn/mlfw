import type { KernelReport, VerifyLevelName } from '../protocol.js';

export const VERIFY_LEVELS: { id: VerifyLevelName; label: string; note: string }[] = [
  {
    id: 'each-pass',
    label: 'each pass',
    note: 'the compiler checks the IR after every pass and stops at the first one that breaks an invariant',
  },
  {
    id: 'boundaries',
    label: 'boundaries',
    note: 'the compiler only checks between phases, so a pass that breaks the IR is still marked here but the pipeline runs on and you can see the damage spread',
  },
  {
    id: 'off',
    label: 'off',
    note: 'nothing is verified — the fastest compile, and the one that lets invalid IR reach codegen',
  },
];

export const TOLERANCES = [1e-6, 1e-4, 1e-3, 1e-2] as const;

export const DEFAULT_TOLERANCE = 1e-3;

export function verifyNote(level: VerifyLevelName): string {
  return (VERIFY_LEVELS.find(entry => entry.id === level) as (typeof VERIFY_LEVELS)[number]).note;
}

type Counted = Extract<{ [K in keyof KernelReport]: KernelReport[K] extends number ? K : never }[keyof KernelReport], string>;

export type KernelMetric = {
  key: Counted;
  label: string;
  meaning: string;
  notable: (report: KernelReport) => boolean;
};

export const KERNEL_METRICS: KernelMetric[] = [
  {
    key: 'bytes',
    label: 'bytes',
    meaning: 'how much source the kernel is. A fused elementwise kernel is a few KB; megabytes means codegen '
      + 'duplicated an expression instead of naming it once.',
    notable: report => report.blownUp,
  },
  {
    key: 'lines',
    label: 'lines',
    meaning: 'statements in the emitted kernel, counted after codegen has done all its inlining.',
    notable: () => false,
  },
  {
    key: 'longestLine',
    label: 'longest line',
    meaning: 'the widest single expression. A line thousands of characters long is a tree that was expanded '
      + 'instead of shared — the shape of a codegen blowup.',
    notable: report => report.blownUp,
  },
  {
    key: 'loops',
    label: 'loops',
    meaning: 'loop nests in the kernel. Fusion should be collapsing these, so a count that tracks the op count '
      + 'means the regions never merged.',
    notable: () => false,
  },
  {
    key: 'tempBuffers',
    label: 'temporaries',
    meaning: 'scratch buffers the kernel allocates for itself. Each one is memory traffic that fusion or the '
      + 'memory planner was supposed to remove.',
    notable: () => false,
  },
  {
    key: 'boundsChecks',
    label: 'bounds checks',
    meaning: 'index guards the analyzer could not prove away. Each is a branch inside the inner loop.',
    notable: report => report.boundsChecks > 0,
  },
  {
    key: 'modulos',
    label: 'modulos',
    meaning: 'integer modulo left in an index expression — the slowest arithmetic there is in an inner loop, and '
      + 'usually a layout the index simplifier could not flatten.',
    notable: report => report.modulos > 0,
  },
];
