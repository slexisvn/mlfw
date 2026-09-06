// RUN: tera-opt %s --pass-pipeline='builtin.module(func.func(tera-tile-and-fuse),canonicalize,cse)' -o %t.pass.mlir
// RUN: tera-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%S/Inputs/cpu-schedule.mlir},transform-interpreter,canonicalize,cse)' -o %t.script.mlir
// RUN: diff %t.pass.mlir %t.script.mlir

// RUN: tera-opt %S/Inputs/gpu-payload.mlir --pass-pipeline='builtin.module(convert-linalg-to-parallel-loops,func.func(tera-tile-parallel-loops),func.func(gpu-map-parallel-loops{mapping-policy=innermost-first}),convert-parallel-loops-to-gpu,func.func(tera-verify-gpu-mapping),canonicalize)' \
// RUN:   | FileCheck %s --check-prefix=LAUNCH
// RUN: tera-opt %S/Inputs/gpu-payload.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%S/Inputs/gpu-schedule.mlir},transform-interpreter,canonicalize)' \
// RUN:   | FileCheck %s --check-prefix=LAUNCH

// Whether a schedule can be written down at all, which is the question every
// tuned schedule depends on and the cheapest one to answer wrongly by
// reasoning. A schedule the autotuner picks has to reach the compiler as data
// -- a transform script it can emit, cache and hand back -- and that is only
// worth building if the vocabulary can already say what the compiler does
// today. So both of the schedules terac has are written twice, once as the
// pass and once as a script, and the two are required to agree.
//
// The CPU pair agree exactly: `Inputs/cpu-schedule.mlir` produces the same
// module `-tera-tile-and-fuse` does, and the test is a `diff` rather than a
// pattern, so a change to either side that the other does not follow is a
// failure rather than a difference nobody looks at. Both sides are cleaned up
// the same way first, because the interpreter leaves a duplicate bound behind
// where the pass folds one and neither is the schedule.
//
// The extents here all divide their tile, so what the two sides are compared
// on is the tiling and the fusion and not the peeling the pass does after
// them; a partial tile is a third thing to write down, and `transform.loop.peel`
// is where it would be written.
//
// The GPU pair agree on the launch and not on the route to it. The pass tiles
// `scf.parallel` after bufferization and lets `convert-parallel-loops-to-gpu`
// read the block shape off the loop nesting; the script tiles the linalg op
// into two `scf.forall` levels and hands the mapping to the GPU transform ops,
// which substitute thread ids into the body rather than building a loop for
// them to be read out of. Different IR on the way, the same grid of the same
// blocks at the end, which is what the schedule is. The check is on the launch
// for that reason.
//
// One thing neither script can do yet is name the op it schedules: the handles
// here are split positionally, so a script belongs to one program rather than
// to a contraction. That is the gap a `tera.schedule` attribute closes, and
// until it is there a tuned script cannot be cached against anything.

// LAUNCH-LABEL: func.func @scale
// LAUNCH-DAG:     %[[COLUMNS:[a-z0-9_]+]] = arith.constant 2 : index
// LAUNCH-DAG:     %[[ROWS:[a-z0-9_]+]] = arith.constant 64 : index
// LAUNCH-DAG:     %[[ONE:[a-z0-9_]+]] = arith.constant 1 : index
// LAUNCH-DAG:     %[[LANES:[a-z0-9_]+]] = arith.constant 256 : index
// LAUNCH:         gpu.launch blocks({{[^)]*}}) in (%{{[a-z0-9_]+}} = %[[COLUMNS]], %{{[a-z0-9_]+}} = %[[ROWS]], %{{[a-z0-9_]+}} = %[[ONE]]) threads({{[^)]*}}) in (%{{[a-z0-9_]+}} = %[[LANES]], %{{[a-z0-9_]+}} = %[[ONE]], %{{[a-z0-9_]+}} = %[[ONE]])
// LAUNCH-NOT:     scf.parallel
// LAUNCH-NOT:     scf.forall

#ewise = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d1)>
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

func.func @fused(%x: tensor<64x256xf32>, %w: tensor<256x128xf32>,
                 %b: tensor<128xf32>) -> tensor<64x128xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x128xf32>
  %filled = linalg.generic {indexing_maps = [#ewise],
                            iterator_types = ["parallel", "parallel"]}
      outs(%empty : tensor<64x128xf32>) {
  ^bb0(%out: f32):
    linalg.yield %zero : f32
  } -> tensor<64x128xf32>
  %product = linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                             iterator_types = ["parallel", "parallel",
                                               "reduction"]}
      ins(%x, %w : tensor<64x256xf32>, tensor<256x128xf32>)
      outs(%filled : tensor<64x128xf32>) {
  ^bb0(%a: f32, %c: f32, %acc: f32):
    %0 = arith.mulf %a, %c : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<64x128xf32>
  %biased = linalg.generic {indexing_maps = [#ewise, #row, #ewise],
                            iterator_types = ["parallel", "parallel"]}
      ins(%product, %b : tensor<64x128xf32>, tensor<128xf32>)
      outs(%empty : tensor<64x128xf32>) {
  ^bb0(%a: f32, %c: f32, %out: f32):
    %0 = arith.addf %a, %c : f32
    %1 = math.exp %0 : f32
    linalg.yield %1 : f32
  } -> tensor<64x128xf32>
  return %biased : tensor<64x128xf32>
}
