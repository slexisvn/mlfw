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

// A loop with no mapping was never a candidate on the host: reductions over
// every axis never become a parallel loop the mapper walks, and nothing asked
// for this one either.

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

// -----

// Inside a kernel the same unmapped loop is the failure this pass exists for.
// `convert-parallel-loops-to-gpu` descends only into a nested `scf.parallel`,
// so one under an `scf.for` is never reached and never annotated: the launch
// it builds is one thread wide and runs the loop's 64 iterations in sequence.
// There is no mapping attribute left behind to notice, which is why the loop
// itself is what is reported.

module attributes {gpu.container_module} {
  func.func @sequential_inside_a_kernel(%out: memref<8x64xf32>, %v: f32) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c8, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      scf.for %k = %c0 to %c8 step %c1 {
        // expected-error @+1 {{is inside a kernel and is still a loop}}
        scf.parallel (%i) = (%c0) to (%c64) step (%c1) {
          memref.store %v, %out[%bx, %i] : memref<8x64xf32>
          scf.reduce
        }
      }
      gpu.terminator
    }
    return
  }
}

// -----

// A loop that reached the device carrying a mapping the conversion could not
// honour is the same failure, and it is reported as the kernel case rather
// than the host one: whatever it still carries, it is running on one thread.

module attributes {gpu.container_module} {
  func.func @mapped_but_left_inside(%out: memref<64xf32>, %v: f32) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      // expected-error @+1 {{is inside a kernel and is still a loop}}
      scf.parallel (%i) = (%c0) to (%c64) step (%c1) {
        memref.store %v, %out[%i] : memref<64xf32>
        scf.reduce
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      gpu.terminator
    }
    return
  }
}
