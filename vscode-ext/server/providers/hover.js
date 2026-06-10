import { wordRangeAt, lineAt } from '../analyzer/position.js';

export const id = 'hover';

export function register(connection, ctx) {
  connection.onHover(params => {
    try {
      return computeHover(ctx, params);
    } catch (err) {
      connection.console.error(`hover error: ${err.message}`);
      return null;
    }
  });
}

function computeHover(ctx, params) {
    const doc = ctx.analyzer.get(params.textDocument.uri);
    if (!doc) return null;
    const word = wordRangeAt(doc.text, params.position);
    if (!word) return null;

    const method = findMethodHover(doc, word, ctx.languageData);
    if (method) return method;

    const builtin = ctx.languageData.builtins.find(b => b.name === word.text);
    if (builtin) {
      const lines = [];
      if (builtin.signature) lines.push('```tera', builtin.signature.display, '```');
      else lines.push(`\`${builtin.name}\``);
      lines.push('', `_${builtin.kind}_`);
      if (builtin.description) lines.push('', builtin.description);
      return { contents: { kind: 'markdown', value: lines.join('\n') }, range: word.range };
    }

    const symbol = doc.symbols.resolve(word.text, params.position);
    if (symbol) {
      const lines = [`\`${symbol.name}\` — *${symbol.kind}*`];
      if (symbol.typeName) lines.push('', `type: \`${symbol.typeName}\``);
      return {
        contents: { kind: 'markdown', value: lines.join('\n') },
        range: word.range,
      };
    }

    if (ctx.languageData.keywords.includes(word.text)) {
      return {
        contents: { kind: 'markdown', value: `\`${word.text}\` — *keyword*` },
        range: word.range,
      };
    }
    return null;
}

function findMethodHover(doc, word, languageData) {
  const line = lineAt(doc.text, word.range.start.line);
  const before = line.slice(0, word.range.start.character);
  const match = before.match(/([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*$/);
  if (!match) return null;
  const receiverName = match[1];
  const symbol = doc.symbols.resolve(receiverName, word.range.start);
  const typeName = symbol?.typeName ?? receiverName;
  const lookup = lookupMethod(typeName, word.text, languageData);
  if (!lookup) return null;
  const lines = [
    '```tera',
    `${lookup.ownerName}.${lookup.method.signature.display}`,
    '```',
    '',
    `_method of ${lookup.ownerName}_`,
  ];
  if (lookup.method.description) lines.push('', lookup.method.description);
  return { contents: { kind: 'markdown', value: lines.join('\n') }, range: word.range };
}

function lookupMethod(typeName, methodName, languageData, seen = new Set()) {
  if (!typeName || seen.has(typeName)) return null;
  seen.add(typeName);
  const builtin = languageData.builtins.find(b => b.name === typeName);
  if (builtin) {
    const own = builtin.methods?.find(m => m.name === methodName);
    if (own) return { ownerName: typeName, method: own };
    if (builtin.returns) return lookupMethod(builtin.returns, methodName, languageData, seen);
    return null;
  }
  const pseudo = languageData.pseudoTypes?.[typeName];
  if (pseudo) {
    const found = pseudo.find(m => m.name === methodName);
    if (found) return { ownerName: typeName, method: found };
  }
  return null;
}
