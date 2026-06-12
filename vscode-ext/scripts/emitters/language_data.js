const SCALAR_KINDS = new Set(['device', 'dtype', 'constant']);

export function buildLanguageData({ keywords, keywordGroups, operators, builtins, pseudoTypes = {} }) {
  return {
    version: 1,
    keywords,
    keywordGroups,
    operators,
    pseudoTypes: serializePseudoTypeMethods(pseudoTypes),
    builtins: builtins.map(b => ({
      name: b.name,
      kind: b.kind,
      description: b.description ?? null,
      returns: b.returns ?? null,
      signature: b.signature ? {
        params: b.signature.params,
        display: formatDisplay(b.name, b.signature.params, b.kind),
      } : null,
      methods: (b.methods ?? []).map(m => ({
        name: m.name,
        description: m.description ?? null,
        returns: m.returns ?? null,
        signature: {
          params: m.params,
          display: formatDisplay(m.name, m.params, b.kind),
        },
      })),
    })),
  };
}

function serializePseudoTypeMethods(types) {
  const out = {};
  for (const [name, entry] of Object.entries(types)) {
    out[name] = (entry.methods ?? []).map(m => ({
      name: m.name,
      description: m.description ?? null,
      returns: m.returns ?? null,
      isGetter: m.isGetter ?? false,
      signature: {
        params: m.params,
        display: m.isGetter ? m.name : formatDisplay(m.name, m.params, 'method'),
      },
    }));
  }
  return out;
}

function formatDisplay(name, params, kind) {
  if (!params.length && SCALAR_KINDS.has(kind)) return name;
  const parts = params.map(p => {
    const prefix = p.rest ? '...' : '';
    if (p.defaultValue !== null && p.defaultValue !== undefined) return `${prefix}${p.name}=${p.defaultValue}`;
    if (p.optional && !p.rest) return `${p.name}?`;
    return `${prefix}${p.name}`;
  });
  return `${name}(${parts.join(', ')})`;
}
