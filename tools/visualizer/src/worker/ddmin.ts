export type Test = (subset: readonly string[]) => Promise<boolean>;

type Reduction = { subset: string[]; parts: number };

function slices(items: readonly string[], parts: number): string[][] {
  const chunks: string[][] = [];
  const size = items.length / parts;

  for (let i = 0; i < parts; i++) {
    const chunk = items.slice(Math.round(i * size), Math.round((i + 1) * size));
    if (chunk.length > 0) chunks.push(chunk);
  }

  return chunks;
}

function without(items: readonly string[], removed: readonly string[]): string[] {
  const drop = new Set(removed);
  return items.filter(item => !drop.has(item));
}

async function reduceOnce(
  current: readonly string[],
  chunks: readonly string[][],
  parts: number,
  test: Test,
): Promise<Reduction | null> {
  for (const chunk of chunks) {
    if (await test(chunk)) return { subset: chunk, parts: 2 };
  }

  for (const chunk of chunks) {
    const rest = without(current, chunk);
    if (rest.length === 0 || rest.length === current.length) continue;
    if (await test(rest)) return { subset: rest, parts: Math.max(parts - 1, 2) };
  }

  return null;
}

export async function ddmin(candidates: readonly string[], test: Test): Promise<string[]> {
  let current = [...candidates];
  let parts = 2;

  while (current.length > 1) {
    const reduced = await reduceOnce(current, slices(current, parts), parts, test);

    if (reduced === null) {
      if (parts >= current.length) break;
      parts = Math.min(current.length, parts * 2);
      continue;
    }

    current = reduced.subset;
    parts = Math.min(reduced.parts, Math.max(current.length, 2));
  }

  return current;
}
