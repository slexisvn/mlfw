// RUN: tera-opt %s --tera-verify-gpu-mapping --split-input-file \
// RUN:   -verify-diagnostics

// A loop the conversion took is a `gpu.launch` by now, so a loop still holding
// the mapping `gpu-map-parallel-loops` gave it is one that asked for a
// processor and did not get one. Left alone it lowers to a sequential host
// loop, which is the right answer computed on one core.

func.func @stayed_on_the_host(%out: memref<64xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  // expected-error @+1 {{carries a processor mapping but is still a loop}}
  scf.parallel (%i) = (%c0) to (%c64) step (%c1) {
    memref.store %v, %out[%i] : memref<64xf32>
    scf.reduce
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return
}

// -----

// A loop with no mapping was never a candidate: reductions over every axis
// never become a parallel loop the mapper walks, and nothing asked for this
// one either.

func.func @never_asked(%out: memref<64xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.parallel (%i) = (%c0) to (%c64) step (%c1) {
    memref.store %v, %out[%i] : memref<64xf32>
    scf.reduce
  }
  return
}

// -----

// The whole point is that the loop is gone once it converted, so a function
// that reached the device has nothing left to report.

module attributes {gpu.container_module} {
  func.func @reached_the_device(%out: memref<64xf32>, %v: f32) {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c64, %sy = %c1, %sz = %c1) {
      memref.store %v, %out[%tx] : memref<64xf32>
      gpu.terminator
    }
    return
  }
}
