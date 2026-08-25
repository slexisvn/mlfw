export const BlockRole = Object.freeze({
  COMPUTE: 'block',
  INIT: 'init',
  ACC: 'acc',
  UPDATE: 'update',
});

export type BlockRoleValue = (typeof BlockRole)[keyof typeof BlockRole];

const ROLE_SUFFIXES: readonly string[] = Object.values(BlockRole).map(role => `_${role}`);

const RUN_SUFFIX = /_\d+$/;

export const CONSTANT_BLOCK_HINT = 'constant_block';

export function numberedBlockName(hint: string, run: number): string {
  return `${hint}_${run}`;
}

export function blockHint(name: string): string {
  return name.replace(RUN_SUFFIX, '');
}

export function opOfBlockName(name: string): string | null {
  const hint = blockHint(name);
  for (const suffix of ROLE_SUFFIXES) {
    if (hint.endsWith(suffix)) return hint.slice(0, -suffix.length) || null;
  }
  return hint || null;
}
