import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ASSETS = resolve(dirname(fileURLToPath(import.meta.url)), '../dist/assets');

const REQUIRED_NODE_TYPES = [
  'PrimFunc', 'ForNode', 'BlockNode', 'SeqNode', 'BufferStoreNode',
  'LIRFunc', 'LIRFlatStoreNode', 'LIRAccumulatorNode',
];

const WHY = `
The compiler derives every IR node's type from its class name (src/compiler/ir/tensor/nodes.ts:24,
"this.type = this.constructor.name"). A minifier that renames classes makes the IR visitor fail at
runtime with "no child schema for node type 'en'" — a break that only appears in a production build,
never in dev. Keep "keepNames: true" on both build.rolldownOptions.output and
worker.rolldownOptions.output in vite.config.ts.

Searching for the bare names is not enough: they also appear as schema keys in ir_visitor, which
minification never touches. What has to survive is the class binding itself.
`.trim();

function declaresClass(source, name) {
  return source.includes(`class ${name} `)
    || source.includes(`class ${name}{`)
    || source.includes(`${name}=class`)
    || source.includes(`${name} = class`);
}

const chunks = readdirSync(ASSETS).filter(name => name.endsWith('.js'));
if (chunks.length === 0) throw new Error(`no javascript chunks in ${ASSETS} — did the build run?`);

const sources = chunks.map(name => readFileSync(join(ASSETS, name), 'utf8'));
const missing = REQUIRED_NODE_TYPES.filter(type => !sources.some(source => declaresClass(source, type)));

if (missing.length > 0) {
  throw new Error(`the build mangled these IR node classes: ${missing.join(', ')}\n\n${WHY}`);
}

console.log(`${chunks.length} chunks checked — all ${REQUIRED_NODE_TYPES.length} IR node classes kept their names`);
