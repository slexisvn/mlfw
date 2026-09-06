// The elementwise op the two GPU schedules in schedules-as-scripts.mlir both
// have to reach the same launch for. It is on buffers because that is where
// the pass being compared against runs -- `-tera-tile-parallel-loops` cuts
// `scf.parallel`, which only exists below bufferization -- and the script
// tiles the same op above it, which is the whole difference between them.

func.func @scale(%x: memref<64x512xf32>, %w: memref<64x512xf32>,
                 %out: memref<64x512xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                   affine_map<(d0, d1) -> (d0, d1)>,
                                   affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
      ins(%x, %w : memref<64x512xf32>, memref<64x512xf32>)
      outs(%out : memref<64x512xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %0 = arith.mulf %a, %b : f32
    linalg.yield %0 : f32
  }
  return
}
