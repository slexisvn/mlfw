import { DiagnosticSeverity } from 'vscode-languageserver';

export const id = 'diagnostics';

export function register(connection, ctx) {
  ctx.bus.on('analyzed', ({ uri, doc }) => {
    const diagnostics = doc.errors.map(toDiagnostic);
    connection.sendDiagnostics({ uri, diagnostics });
  });
  ctx.bus.on('closed', ({ uri }) => {
    connection.sendDiagnostics({ uri, diagnostics: [] });
  });
}

function toDiagnostic(err) {
  const line = Math.max(0, (err.line ?? 1) - 1);
  const character = Math.max(0, (err.column ?? 1) - 1);
  return {
    severity: DiagnosticSeverity.Error,
    range: {
      start: { line, character },
      end: { line, character: character + 1 },
    },
    message: err.message,
    source: `tera:${err.source ?? 'parser'}`,
  };
}
