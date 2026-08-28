import type { Kernel, NestNode, Snapshot } from '../protocol.js';

const LABEL_LIMIT = 88;
const CLOSER_ONLY = /^[)\]}]+[;,]?$/;

function indentOf(line: string): number {
  let width = 0;
  for (const char of line) {
    if (char === ' ') width += 1;
    else if (char === '\t') width += 2;
    else break;
  }
  return width;
}

export function nestForSource(kernel: Kernel): NestNode {
  const root: NestNode = {
    id: `source:${kernel.name}`,
    kind: 'source',
    label: kernel.name,
    detail: kernel.language,
    op: null,
    opId: null,
    line: null,
    children: [],
  };

  const stack: { indent: number; node: NestNode }[] = [{ indent: -1, node: root }];
  const lines = kernel.source.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const text = lines[i].trim();
    if (text === '' || CLOSER_ONLY.test(text)) continue;

    const indent = indentOf(lines[i]);
    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) stack.pop();

    const node: NestNode = {
      id: `line:${kernel.name}:${i}`,
      kind: 'line',
      label: text.length > LABEL_LIMIT ? `${text.slice(0, LABEL_LIMIT)}…` : text,
      detail: '',
      op: null,
      opId: null,
      line: null,
      children: [],
    };

    stack[stack.length - 1].node.children.push(node);
    stack.push({ indent, node });
  }

  return root;
}

export function sourceSnapshot(kernels: readonly Kernel[]): Snapshot {
  return {
    text: kernels.map(k => k.source).join('\n\n'),
    bytes: 0, flops: 0, ops: kernels.reduce((total, k) => total + k.source.split('\n').filter(l => l.trim() !== '').length, 0),
    dags: [],
    nests: kernels.map(nestForSource),
  };
}
