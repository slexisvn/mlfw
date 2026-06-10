import { SemanticTokensBuilder } from 'vscode-languageserver';

export const id = 'semanticTokens';

export const legend = {
  tokenTypes: [
    'namespace', 'class', 'enumMember', 'parameter', 'variable',
    'function', 'keyword',
  ],
  tokenModifiers: ['declaration'],
};

const KIND_TO_TYPE = {
  model: 'class',
  function: 'function',
  parameter: 'parameter',
  variable: 'variable',
  module: 'class',
  optimizer: 'class',
  scheduler: 'class',
  metric: 'class',
  callback: 'class',
  logger: 'class',
  trainer: 'class',
  sequential: 'class',
  device: 'enumMember',
  dtype: 'enumMember',
  factory: 'function',
  reduction: 'function',
  utility: 'function',
  shape: 'function',
  autograd: 'function',
  data: 'function',
  constant: 'enumMember',
};

export function register(connection, ctx) {
  connection.languages.semanticTokens.on(params => {
    const doc = ctx.analyzer.get(params.textDocument.uri);
    if (!doc) return { data: [] };
    return buildTokens(doc, ctx.languageData, legend);
  });
}

function buildTokens(doc, languageData, legend) {
  const typeIndex = new Map(legend.tokenTypes.map((t, i) => [t, i]));
  const builder = new SemanticTokensBuilder();
  const builtinByName = new Map(languageData.builtins.map(b => [b.name, b]));
  const symbolByName = new Map();
  for (const s of doc.symbols.flat) symbolByName.set(s.name, s);

  const tokens = doc.tokens ?? [];
  for (const tok of tokens) {
    if (tok.type !== 'identifier') continue;
    const builtin = builtinByName.get(tok.value);
    const type = builtin
      ? KIND_TO_TYPE[builtin.kind]
      : KIND_TO_TYPE[symbolByName.get(tok.value)?.kind];
    if (!type) continue;
    const line = Math.max(0, tok.line - 1);
    const character = Math.max(0, tok.column - 1);
    builder.push(line, character, tok.value.length, typeIndex.get(type), 0);
  }
  return builder.build();
}
