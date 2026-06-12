import { mkdirSync, writeFileSync, copyFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractKeywords, classifyKeywords } from './extractors/keywords.js';
import { extractOperators } from './extractors/operators.js';
import { extractBuiltins } from './extractors/builtins.js';
import { extractBuiltinDocs } from './extractors/builtin_docs.js';
import { buildGrammar } from './emitters/grammar.js';
import { buildLanguageData } from './emitters/language_data.js';
import { buildSnippets } from './emitters/snippets.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const EXT_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(EXT_ROOT, '..');

const SOURCES = {
  parser: join(REPO_ROOT, 'src/cli/parser.js'),
  tokenizer: join(REPO_ROOT, 'src/cli/tokenizer.js'),
  builtins: join(REPO_ROOT, 'src/cli/builtins.js'),
  builtinDocs: join(EXT_ROOT, 'data/builtin-docs.md'),
};

const OUTPUTS = {
  grammar: join(EXT_ROOT, 'syntaxes/tera.tmLanguage.json'),
  languageData: join(EXT_ROOT, 'language-data.json'),
  snippets: join(EXT_ROOT, 'snippets/tera.json'),
  vendorDir: join(EXT_ROOT, 'server/vendor'),
};

const VENDORED_FILES = ['tokenizer.js', 'parser.js'];

export function generate(sources = SOURCES, outputs = OUTPUTS) {
  const keywords = extractKeywords(sources.parser);
  const keywordGroups = classifyKeywords(keywords);
  const operators = extractOperators(sources.tokenizer);
  const builtins = extractBuiltins(sources.builtins);
  const docs = extractBuiltinDocs(sources.builtinDocs);

  const enriched = builtins.map(b => mergeDoc(b, docs.builtins.get(b.name), docs.kindTemplates));
  const undocumented = enriched.filter(b => !b.documented).map(b => b.name);

  const runtimeNames = new Set(builtins.map(b => b.name));
  const docOnly = [];
  for (const [name, doc] of docs.builtins) {
    if (runtimeNames.has(name)) continue;
    docOnly.push(buildDocOnlyBuiltin(doc, docs.kindTemplates));
  }
  const allBuiltins = [...enriched, ...docOnly];

  const pseudoTypes = serializePseudoTypes(docs.pseudoTypes);

  const grammar = buildGrammar({ keywordGroups, operators, builtins: allBuiltins });
  const languageData = buildLanguageData({ keywords, keywordGroups, operators, builtins: allBuiltins, pseudoTypes });
  const snippets = buildSnippets({ builtins: allBuiltins });

  ensureDir(dirname(outputs.grammar));
  ensureDir(dirname(outputs.snippets));
  writeJson(outputs.grammar, grammar);
  writeJson(outputs.languageData, languageData);
  writeJson(outputs.snippets, snippets);
  if (outputs.vendorDir) vendorSources(join(REPO_ROOT, 'src/cli'), outputs.vendorDir);

  return { keywords, operators, builtins: enriched, undocumented };
}

const RETURNS_BY_KIND = {
  factory: 'Tensor',
  function: 'Tensor',
  reduction: 'Tensor',
  shape: 'Tensor',
  autograd: 'Tensor',
};

const RETURNS_OVERRIDE = {
  shape: null,
  dtype: null,
  len: null,
  range: null,
  print: null,
  trace: null,
  graph: null,
  compile: null,
  load_csv: 'DataFrame',
  normalize: 'Tensor',
  encode: null,
  decode: null,
  train_test_split: null,
  DataFrame: 'DataFrame',
  col: 'Column',
  lit: 'Column',
  expr: 'Column',
  avg: 'Column',
  count: 'Column',
  countStar: 'Column',
};

const DEFAULT_DOC_BUILTIN_KIND = 'function';

function inferReturns(builtin) {
  if (builtin.name in RETURNS_OVERRIDE) return RETURNS_OVERRIDE[builtin.name];
  return RETURNS_BY_KIND[builtin.kind] ?? null;
}

function mergeDoc(builtin, doc, kindTemplates) {
  const template = kindTemplates.get(builtin.kind);
  const returns = inferReturns(builtin);
  if (!doc) {
    return {
      ...builtin,
      signature: null,
      description: null,
      methods: template ? [...template.methods] : [],
      returns,
      documented: false,
    };
  }
  const ownMethodNames = new Set((doc.methods ?? []).map(m => m.name));
  const inherited = (template?.methods ?? []).filter(m => !ownMethodNames.has(m.name));
  // A doc-declared `{kind}` annotation overrides the kind inferred from source.
  const kind = doc.kind ?? builtin.kind;
  return {
    ...builtin,
    kind,
    description: doc.description,
    signature: doc.params === null ? null : { params: doc.params },
    methods: [...(doc.methods ?? []), ...inherited],
    returns,
    documented: true,
  };
}

function buildDocOnlyBuiltin(doc, kindTemplates) {
  const kind = doc.kind ?? DEFAULT_DOC_BUILTIN_KIND;
  const template = kindTemplates.get(kind);
  const ownMethodNames = new Set((doc.methods ?? []).map(m => m.name));
  const inherited = (template?.methods ?? []).filter(m => !ownMethodNames.has(m.name));
  return {
    name: doc.name,
    kind,
    description: doc.description,
    signature: doc.params === null ? null : { params: doc.params },
    methods: [...(doc.methods ?? []), ...inherited],
    returns: inferReturns({ name: doc.name, kind }),
    documented: true,
  };
}

function serializePseudoTypes(map) {
  const out = {};
  for (const [name, entry] of map) out[name] = { methods: entry.methods };
  return out;
}

function vendorSources(srcDir, destDir) {
  ensureDir(destDir);
  for (const file of VENDORED_FILES) copyFileSync(join(srcDir, file), join(destDir, file));
}

function ensureDir(path) {
  mkdirSync(path, { recursive: true });
}

function writeJson(path, value) {
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n', 'utf8');
}

const isEntry = process.argv[1] && process.argv[1].endsWith('generate.js');
if (isEntry) {
  const result = generate();
  process.stdout.write(`Generated: ${result.builtins.length} builtins, ${result.keywords.length} keywords\n`);
  if (result.undocumented.length) {
    process.stdout.write(`Undocumented (${result.undocumented.length}): ${result.undocumented.join(', ')}\n`);
  }
}
