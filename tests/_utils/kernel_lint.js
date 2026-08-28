const JS_ARTIFACTS = ['undefined', '[object', '=>', 'Math.', 'function ', 'NaN'];

function checkBalanced(src, open, close) {
  let depth = 0;
  for (let i = 0; i < src.length; i++) {
    if (src[i] === open) depth++;
    else if (src[i] === close) { depth--; if (depth < 0) return false; }
  }
  return depth === 0;
}

function stripStringsAndComments(src) {
  return src
    .replace(/\/\/[^\n]*/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/"(?:[^"\\]|\\.)*"/g, '""');
}

function detectLanguage(src) {
  if (/@compute|@workgroup_size|var<|fn\s+\w+\s*\(/.test(src)) return 'wgsl';
  if (/__global__|__shared__|blockIdx|threadIdx/.test(src)) return 'cuda';
  if (/\(module|\(func|local\.get|i32\.|f32\.load/.test(src)) return 'wat';
  if (/\bfunction\b|=>|\blet\b|\bconst\b\s+\w+\s*=/.test(src)) return 'js';
  return 'c';
}

const JS_ARTIFACTS_ONLY = ['undefined', '[object'];

function findJsBackendArtifacts(src) {
  const found = [];
  for (const a of JS_ARTIFACTS_ONLY) if (src.includes(a)) found.push(a);
  if (/(?<![.\w])NaN(?![\w])/.test(src)) found.push('NaN');
  return found;
}

function findJsArtifacts(src) {
  const found = [];
  for (const a of JS_ARTIFACTS) {
    if (a === 'NaN') {
      if (/\bNaN\b/.test(src)) found.push(a);
    } else if (src.includes(a)) {
      found.push(a);
    }
  }
  return found;
}

function findSelfAssign(src) {
  const out = [];
  for (const m of src.matchAll(/([A-Za-z_]\w*\[[^\]]+\])\s*=\s*([^;=][^;]*);/g)) {
    if (m[2].includes('==')) continue;
    if (m[1].replace(/\s+/g, '') === m[2].replace(/\s+/g, '')) out.push(m[0].trim());
  }
  return out;
}

function findLiteralDivByZero(src) {
  return /\/\s*0(?:\.0+)?(?![.0-9eExXa-fA-F])/.test(src);
}

const C_KEYWORDS = new Set([
  'const', 'int', 'float', 'double', 'void', 'for', 'if', 'else', 'while', 'return',
  'unsigned', 'long', 'short', 'char', 'bool', 'true', 'false', 'sizeof', 'struct',
  'static', 'extern', 'volatile', 'restrict', 'inline',
  '__global__', '__shared__', '__device__', '__host__', '__restrict__', '__half', '__forceinline__',
  '__syncthreads', 'blockIdx', 'threadIdx', 'blockDim', 'gridDim', 'INFINITY', 'NAN',
  'pragma', 'unroll', 'alloca',
]);

const C_BUILTINS = /^(expf?|logf?|log2f?|log10f?|exp2f?|sqrtf?|rsqrtf?|tanhf?|sinhf?|coshf?|fabsf?|sinf?|cosf?|tanf?|ceilf?|floorf?|fmaxf?|fminf?|powf?|roundf?|fmodf?|erff?|atanf?|asinf?|acosf?|copysignf?|isnan|isinf|min|max|abs|fma|fmaf|__expf|__logf|make_float4|atomicAdd)$/;

function findUndeclaredC(src) {
  const declared = new Set(C_KEYWORDS);
  for (const m of src.matchAll(/(?:const\s+)?(?:unsigned\s+)?(?:int|float|double|__half|bool|long|short|char|size_t)\s+\*?\s*(\w+)\s*(?:=|\[|;|,|\))/g)) {
    declared.add(m[1]);
  }
  for (const m of src.matchAll(/for\s*\(\s*(?:const\s+)?(?:int|unsigned|long|size_t)\s+(\w+)/g)) declared.add(m[1]);
  for (const m of src.matchAll(/\)\s*\{/g)) { /* fn boundaries ignored */ }
  for (const m of src.matchAll(/(?:__global__|__device__|void|int|float)\s+\w+\s*\(([^)]*)\)/g)) {
    for (const p of m[1].split(',')) {
      const pm = p.trim().match(/(\w+)\s*$/);
      if (pm) declared.add(pm[1]);
    }
  }
  const body = src.replace(/^[\s\S]*?\{/, '');
  const used = new Set();
  for (const m of body.matchAll(/\b([A-Za-z_]\w*)\b(?!\s*\()/g)) {
    const n = m[1];
    if (/^\d/.test(n)) continue;
    if (C_BUILTINS.test(n)) continue;
    if (m.index > 0 && body[m.index - 1] === '.') continue;
    used.add(n);
  }
  const undeclared = [];
  for (const v of used) if (!declared.has(v) && !C_BUILTINS.test(v)) undeclared.push(v);
  return undeclared;
}

function checkGpuBoundsGuard(src) {
  const usesId = /blockIdx|threadIdx|global_id|global_invocation_id/.test(src);
  if (!usesId) return true;
  const hasGuard = /if\s*\([^)]*[<>]=?[^)]*\)/.test(src) || /return\s*;/.test(src);
  return hasGuard;
}

function checkWgslBarrier(src) {
  const writesShared = /var<workgroup>/.test(src) && /\bworkgroup\w*\s*\[[^\]]*\]\s*=/.test(src);
  if (!writesShared) return true;
  return /workgroupBarrier\s*\(/.test(src);
}

export function lintKernel(src, opts = {}) {
  const issues = [];
  const lang = opts.lang || detectLanguage(src);
  const clean = stripStringsAndComments(src);

  const artifacts = lang === 'js' ? findJsBackendArtifacts(clean) : findJsArtifacts(clean);
  for (const a of artifacts) issues.push({ kind: 'js_artifact', detail: a });

  if (!checkBalanced(clean, '{', '}')) issues.push({ kind: 'unbalanced', detail: '{}' });
  if (!checkBalanced(clean, '(', ')')) issues.push({ kind: 'unbalanced', detail: '()' });
  if (lang !== 'wat' && !checkBalanced(clean, '[', ']')) issues.push({ kind: 'unbalanced', detail: '[]' });

  for (const s of findSelfAssign(clean)) issues.push({ kind: 'self_assign', detail: s });

  if (findLiteralDivByZero(clean)) issues.push({ kind: 'div_by_zero', detail: '/0' });

  if (lang === 'cuda' || lang === 'c') {
    for (const v of findUndeclaredC(clean)) issues.push({ kind: 'undeclared', detail: v });
    if (!checkGpuBoundsGuard(clean)) issues.push({ kind: 'missing_bounds_guard', detail: lang });
  }

  if (lang === 'wgsl') {
    if (!checkGpuBoundsGuard(clean)) issues.push({ kind: 'missing_bounds_guard', detail: 'wgsl' });
    if (!checkWgslBarrier(clean)) issues.push({ kind: 'missing_barrier', detail: 'wgsl shared write without workgroupBarrier' });
  }

  return { lang, issues };
}

const STRICT_KINDS = new Set(['js_artifact', 'unbalanced', 'self_assign', 'undeclared', 'div_by_zero']);

export function lintKernelStrict(src, opts = {}) {
  const { issues } = lintKernel(src, opts);
  return issues.filter(i => STRICT_KINDS.has(i.kind));
}
