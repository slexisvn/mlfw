export const TargetAttr = Object.freeze({
  GRAPH_SPLIT: 'graphSplit',
  SCHEDULING: 'scheduling',
  FUSION: 'fusion',
  EXTERNAL_COMPILER: 'externalCompiler',
  NATIVE_OPS: 'nativeOps',
});

export type GraphSplitThresholds = Readonly<Record<string, number>>;
export type SchedulingDefaults = Readonly<{ enabled?: boolean; autotune?: boolean; gpuTiling?: boolean }>;
export type FusionDefaults = Readonly<{ enabled?: boolean; strategy?: string }>;

export type AttributedTarget = { getAttr?<T>(key: string, fallback?: T | null): T | null };

export function targetAttr<T>(target: AttributedTarget | null | undefined, key: string): T | null {
  if (!target || typeof target.getAttr !== 'function') return null;
  return target.getAttr<T>(key);
}
