import { DiagnosticSeverity } from 'vscode-languageserver';

export const id = 'diagnostics';

export function register(connection, ctx) {
  ctx.bus.on('analyzed', ({ uri, doc }) => {
    const diagnostics = doc.errors.map(err => toDiagnostic(err, doc));
    connection.sendDiagnostics({ uri, diagnostics });
  });
  ctx.bus.on('closed', ({ uri }) => {
    connection.sendDiagnostics({ uri, diagnostics: [] });
  });
}

export function toDiagnostic(err, doc) {
  const line = Math.max(0, (err.line ?? 1) - 1);
  const character = Math.max(0, (err.column ?? 1) - 1);
  const end = spanEnd(doc, err.line ?? 1, err.column ?? 1) ?? { line, character: character + 1 };
  return {
    severity: DiagnosticSeverity.Error,
    range: { start: { line, character }, end },
    message: err.message,
    source: `tera:${err.source ?? 'parser'}`,
  };
}

function spanEnd(doc, line, column) {
  const token = doc?.tokens?.find(t =>
    t.line === line && t.column === column && t.endColumn !== undefined && t.type !== 'newline');
  if (!token) return null;
  return { line: (token.endLine ?? token.line) - 1, character: token.endColumn - 1 };
}
