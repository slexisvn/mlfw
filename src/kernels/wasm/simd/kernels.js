import { F64_BYTES, I32_BYTES, F64_LANES, LANE_SHIFT } from './runtime.js';

const DIST_SENTINEL = 1e308;
const MEM_MIN_PAGES = 1;
const MEM_MAX_PAGES = 32768;

function moduleWrap(exportName, params, locals, body) {
  const paramDecls = params.map(([n, t]) => `(param $${n} ${t})`).join(' ');
  const localDecls = locals.map(([n, t]) => `(local $${n} ${t})`).join(' ');
  return `(module (memory (export "memory") ${MEM_MIN_PAGES} ${MEM_MAX_PAGES})
 (func (export "${exportName}") ${paramDecls}
  ${localDecls}
${body}))`;
}

function sqDistSnippet(tag, xrow, crow, d, out, t) {
  return `
    (local.get $${d}) (i32.const ${LANE_SHIFT}) i32.shr_s (i32.const ${LANE_SHIFT}) i32.shl local.set $${t.nmain}
    (f64.const 0) f64x2.splat local.set $${t.acc}
    (i32.const 0) local.set $${t.j}
    (block $${tag}_ve (loop $${tag}_vl
      (local.get $${t.j}) (local.get $${t.nmain}) i32.ge_s br_if $${tag}_ve
      (local.get $${xrow}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add v128.load local.set $${t.va}
      (local.get $${crow}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add v128.load local.set $${t.vb}
      (local.get $${t.acc})
      (local.get $${t.va}) (local.get $${t.vb}) f64x2.sub local.set $${t.va}
      (local.get $${t.va}) (local.get $${t.va}) f64x2.mul
      f64x2.add local.set $${t.acc}
      (local.get $${t.j}) (i32.const ${F64_LANES}) i32.add local.set $${t.j}
      br $${tag}_vl))
    (local.get $${t.acc}) f64x2.extract_lane 0
    (local.get $${t.acc}) f64x2.extract_lane 1
    f64.add local.set $${out}
    (block $${tag}_te (loop $${tag}_tl
      (local.get $${t.j}) (local.get $${d}) i32.ge_s br_if $${tag}_te
      (local.get $${xrow}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add f64.load
      (local.get $${crow}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add f64.load
      f64.sub local.set $${t.diff}
      (local.get $${out}) (local.get $${t.diff}) (local.get $${t.diff}) f64.mul f64.add local.set $${out}
      (local.get $${t.j}) (i32.const 1) i32.add local.set $${t.j}
      br $${tag}_tl))`;
}

const _DIST_TMP = { j: 'j', nmain: 'nmain', acc: 'acc', va: 'va', vb: 'vb', diff: 'diff' };
const _DIST_LOCALS = [
  ['j', 'i32'], ['nmain', 'i32'], ['acc', 'v128'], ['va', 'v128'], ['vb', 'v128'], ['diff', 'f64'],
];

export function nearestCentroidWat() {
  const body = `
    (f64.const 0) local.set $inertiaAcc
    (i32.const 0) local.set $i
    (block $iend (loop $iloop
      (local.get $i) (local.get $n) i32.ge_s br_if $iend
      (local.get $x) (local.get $i) (local.get $d) i32.mul (i32.const ${F64_BYTES}) i32.mul i32.add local.set $xrow
      (i32.const 0) local.set $best
      (f64.const ${DIST_SENTINEL}) local.set $bestd
      (i32.const 0) local.set $cc
      (block $cend (loop $cloop
        (local.get $cc) (local.get $k) i32.ge_s br_if $cend
        (local.get $c) (local.get $cc) (local.get $d) i32.mul (i32.const ${F64_BYTES}) i32.mul i32.add local.set $crow
${sqDistSnippet('nc', 'xrow', 'crow', 'd', 'dist', _DIST_TMP)}
        (local.get $dist) (local.get $bestd) f64.lt
        (if (then
          (local.get $dist) local.set $bestd
          (local.get $cc) local.set $best))
        (local.get $cc) (i32.const 1) i32.add local.set $cc
        br $cloop))
      (local.get $labels) (local.get $i) (i32.const ${I32_BYTES}) i32.mul i32.add (local.get $best) i32.store
      (local.get $inertiaAcc) (local.get $bestd) f64.add local.set $inertiaAcc
      (local.get $i) (i32.const 1) i32.add local.set $i
      br $iloop))
    (local.get $inertia) (local.get $inertiaAcc) f64.store`;
  return moduleWrap(
    'nearest_centroid',
    [['x', 'i32'], ['n', 'i32'], ['d', 'i32'], ['c', 'i32'], ['k', 'i32'], ['labels', 'i32'], ['inertia', 'i32']],
    [
      ['i', 'i32'], ['cc', 'i32'], ['best', 'i32'], ['bestd', 'f64'], ['dist', 'f64'],
      ['xrow', 'i32'], ['crow', 'i32'], ['inertiaAcc', 'f64'], ..._DIST_LOCALS,
    ],
    body,
  );
}

export function distRowWat() {
  const body = `
    (i32.const 0) local.set $t
    (block $tend (loop $tloop
      (local.get $t) (local.get $ntr) i32.ge_s br_if $tend
      (local.get $tr) (local.get $t) (local.get $d) i32.mul (i32.const ${F64_BYTES}) i32.mul i32.add local.set $trow
${sqDistSnippet('dr', 'q', 'trow', 'd', 'dist', _DIST_TMP)}
      (local.get $out) (local.get $t) (i32.const ${F64_BYTES}) i32.mul i32.add (local.get $dist) f64.store
      (local.get $t) (i32.const 1) i32.add local.set $t
      br $tloop))`;
  return moduleWrap(
    'dist_row',
    [['q', 'i32'], ['tr', 'i32'], ['ntr', 'i32'], ['d', 'i32'], ['out', 'i32']],
    [['t', 'i32'], ['trow', 'i32'], ['dist', 'f64'], ..._DIST_LOCALS],
    body,
  );
}

function reduceDotSnippet(tag, a, b, n, out, t) {
  return `
    (local.get $${n}) (i32.const ${LANE_SHIFT}) i32.shr_s (i32.const ${LANE_SHIFT}) i32.shl local.set $${t.nmain}
    (f64.const 0) f64x2.splat local.set $${t.acc}
    (i32.const 0) local.set $${t.j}
    (block $${tag}_ve (loop $${tag}_vl
      (local.get $${t.j}) (local.get $${t.nmain}) i32.ge_s br_if $${tag}_ve
      (local.get $${t.acc})
      (local.get $${a}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add v128.load
      (local.get $${b}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add v128.load
      f64x2.mul f64x2.add local.set $${t.acc}
      (local.get $${t.j}) (i32.const ${F64_LANES}) i32.add local.set $${t.j}
      br $${tag}_vl))
    (local.get $${t.acc}) f64x2.extract_lane 0
    (local.get $${t.acc}) f64x2.extract_lane 1
    f64.add local.set $${out}
    (block $${tag}_te (loop $${tag}_tl
      (local.get $${t.j}) (local.get $${n}) i32.ge_s br_if $${tag}_te
      (local.get $${out})
      (local.get $${a}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add f64.load
      (local.get $${b}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add f64.load
      f64.mul f64.add local.set $${out}
      (local.get $${t.j}) (i32.const 1) i32.add local.set $${t.j}
      br $${tag}_tl))`;
}

function axpySnippet(tag, y, x, scalar, n, t) {
  return `
    (local.get $${scalar}) f64x2.splat local.set $${t.sv}
    (local.get $${n}) (i32.const ${LANE_SHIFT}) i32.shr_s (i32.const ${LANE_SHIFT}) i32.shl local.set $${t.nmain}
    (i32.const 0) local.set $${t.j}
    (block $${tag}_ve (loop $${tag}_vl
      (local.get $${t.j}) (local.get $${t.nmain}) i32.ge_s br_if $${tag}_ve
      (local.get $${y}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add local.set $${t.ya}
      (local.get $${t.ya}) v128.load
      (local.get $${t.sv})
      (local.get $${x}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add v128.load
      f64x2.mul f64x2.sub local.set $${t.vy}
      (local.get $${t.ya}) (local.get $${t.vy}) v128.store
      (local.get $${t.j}) (i32.const ${F64_LANES}) i32.add local.set $${t.j}
      br $${tag}_vl))
    (block $${tag}_te (loop $${tag}_tl
      (local.get $${t.j}) (local.get $${n}) i32.ge_s br_if $${tag}_te
      (local.get $${y}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add local.set $${t.ya}
      (local.get $${t.ya})
      (local.get $${t.ya}) f64.load
      (local.get $${scalar})
      (local.get $${x}) (local.get $${t.j}) (i32.const ${F64_BYTES}) i32.mul i32.add f64.load
      f64.mul f64.sub f64.store
      (local.get $${t.j}) (i32.const 1) i32.add local.set $${t.j}
      br $${tag}_tl))`;
}

export function gramSymWat() {
  const dot = { j: 'kk', nmain: 'nm', acc: 'acc' };
  const body = `
    (i32.const 0) local.set $i
    (block $ie (loop $il
      (local.get $i) (local.get $m) i32.ge_s br_if $ie
      (local.get $mat) (local.get $i) (local.get $len) i32.mul (i32.const ${F64_BYTES}) i32.mul i32.add local.set $rowI
      (local.get $i) local.set $j
      (block $je (loop $jl
        (local.get $j) (local.get $m) i32.ge_s br_if $je
        (local.get $mat) (local.get $j) (local.get $len) i32.mul (i32.const ${F64_BYTES}) i32.mul i32.add local.set $rowJ
${reduceDotSnippet('gs', 'rowI', 'rowJ', 'len', 'd', dot)}
        (local.get $g) (local.get $i) (local.get $m) i32.mul (local.get $j) i32.add (i32.const ${F64_BYTES}) i32.mul i32.add (local.get $d) f64.store
        (local.get $g) (local.get $j) (local.get $m) i32.mul (local.get $i) i32.add (i32.const ${F64_BYTES}) i32.mul i32.add (local.get $d) f64.store
        (local.get $j) (i32.const 1) i32.add local.set $j
        br $jl))
      (local.get $i) (i32.const 1) i32.add local.set $i
      br $il))`;
  return moduleWrap(
    'gram_sym',
    [['mat', 'i32'], ['m', 'i32'], ['len', 'i32'], ['g', 'i32']],
    [
      ['i', 'i32'], ['j', 'i32'], ['rowI', 'i32'], ['rowJ', 'i32'], ['d', 'f64'],
      ['kk', 'i32'], ['nm', 'i32'], ['acc', 'v128'],
    ],
    body,
  );
}

export function matmulRowsWat() {
  const dot = { j: 'kk', nmain: 'nm', acc: 'acc' };
  const body = `
    (i32.const 0) local.set $i
    (block $ie (loop $il
      (local.get $i) (local.get $m) i32.ge_s br_if $ie
      (local.get $a) (local.get $i) (local.get $len) i32.mul (i32.const ${F64_BYTES}) i32.mul i32.add local.set $rowA
      (i32.const 0) local.set $c
      (block $ce (loop $cl
        (local.get $c) (local.get $p) i32.ge_s br_if $ce
        (local.get $b) (local.get $c) (local.get $len) i32.mul (i32.const ${F64_BYTES}) i32.mul i32.add local.set $rowB
${reduceDotSnippet('mm', 'rowA', 'rowB', 'len', 'd', dot)}
        (local.get $out) (local.get $i) (local.get $p) i32.mul (local.get $c) i32.add (i32.const ${F64_BYTES}) i32.mul i32.add (local.get $d) f64.store
        (local.get $c) (i32.const 1) i32.add local.set $c
        br $cl))
      (local.get $i) (i32.const 1) i32.add local.set $i
      br $il))`;
  return moduleWrap(
    'matmul_rows',
    [['a', 'i32'], ['m', 'i32'], ['len', 'i32'], ['b', 'i32'], ['p', 'i32'], ['out', 'i32']],
    [
      ['i', 'i32'], ['c', 'i32'], ['rowA', 'i32'], ['rowB', 'i32'], ['d', 'f64'],
      ['kk', 'i32'], ['nm', 'i32'], ['acc', 'v128'],
    ],
    body,
  );
}

export function coordDescentWat() {
  const dot = { j: 'j', nmain: 'nmain', acc: 'acc' };
  const ax = { j: 'j2', nmain: 'nmain2', sv: 'sv', ya: 'ya', vy: 'vy' };
  const body = `
    (local.get $n) f64.convert_i32_s local.set $nf
    (i32.const 0) local.set $iter
    (block $ie (loop $il
      (local.get $iter) (local.get $maxIter) i32.ge_s br_if $ie
      (f64.const 0) local.set $maxChange
      (i32.const 0) local.set $jc
      (block $je (loop $jl
        (local.get $jc) (local.get $d) i32.ge_s br_if $je
        (local.get $z) (local.get $jc) (i32.const ${F64_BYTES}) i32.mul i32.add f64.load local.set $zj
        (local.get $zj) (f64.const 0) f64.ne
        (if (then
          (local.get $xc) (local.get $jc) (local.get $n) i32.mul (i32.const ${F64_BYTES}) i32.mul i32.add local.set $col
${reduceDotSnippet('cd', 'col', 'r', 'n', 'dot', dot)}
          (local.get $w) (local.get $jc) (i32.const ${F64_BYTES}) i32.mul i32.add f64.load local.set $wold
          (local.get $dot) (local.get $nf) f64.div (local.get $wold) (local.get $zj) f64.mul f64.add local.set $rho
          (local.get $rho) (local.get $l1) f64.gt
          (if (then
            (local.get $rho) (local.get $l1) f64.sub local.set $thr
          ) (else
            (local.get $rho) (f64.const 0) (local.get $l1) f64.sub f64.lt
            (if (then
              (local.get $rho) (local.get $l1) f64.add local.set $thr
            ) (else
              (f64.const 0) local.set $thr))))
          (local.get $thr) (local.get $zj) (local.get $l2) f64.add f64.div local.set $wj
          (local.get $wj) (local.get $wold) f64.sub local.set $delta
          (local.get $delta) (f64.const 0) f64.ne
          (if (then
${axpySnippet('cd', 'r', 'col', 'delta', 'n', ax)}
            (local.get $w) (local.get $jc) (i32.const ${F64_BYTES}) i32.mul i32.add (local.get $wj) f64.store
            (local.get $delta) f64.abs local.set $absd
            (local.get $absd) (local.get $maxChange) f64.gt
            (if (then (local.get $absd) local.set $maxChange))))))
        (local.get $jc) (i32.const 1) i32.add local.set $jc
        br $jl))
      (local.get $maxChange) (local.get $tol) f64.lt br_if $ie
      (local.get $iter) (i32.const 1) i32.add local.set $iter
      br $il))`;
  return moduleWrap(
    'coord_descent',
    [
      ['xc', 'i32'], ['n', 'i32'], ['d', 'i32'], ['r', 'i32'], ['w', 'i32'], ['z', 'i32'],
      ['l1', 'f64'], ['l2', 'f64'], ['maxIter', 'i32'], ['tol', 'f64'],
    ],
    [
      ['iter', 'i32'], ['jc', 'i32'], ['nf', 'f64'], ['maxChange', 'f64'], ['zj', 'f64'],
      ['col', 'i32'], ['dot', 'f64'], ['wold', 'f64'], ['rho', 'f64'], ['thr', 'f64'],
      ['wj', 'f64'], ['delta', 'f64'], ['absd', 'f64'],
      ['j', 'i32'], ['nmain', 'i32'], ['acc', 'v128'],
      ['j2', 'i32'], ['nmain2', 'i32'], ['sv', 'v128'], ['ya', 'i32'], ['vy', 'v128'],
    ],
    body,
  );
}
