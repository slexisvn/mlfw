// RUN: tera-opt %s --split-input-file --verify-diagnostics --tera-stage-gpu-buffers

module attributes {gpu.container_module} {
  // The buffer lives on the device from the copy in to the copy back, so a host
  // read on either side of that window is fine and one inside it is not. A
  // reduction is where this nearly happens for real -- it has no parallel axis
  // left to map and stays on the host -- and there it reads before or after
  // every kernel, which is allowed. Reading between two of them is not, and is
  // reported rather than silently mislowered.

  func.func @host_read_between_kernels() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<4xf32>
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<4xf32>, %alloc : memref<4xf32>)
    // expected-error @+1 {{only current copy on the device}}
    %v = memref.load %alloc[%c0] : memref<4xf32>
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<4xf32>, %alloc : memref<4xf32>)
    return %v : f32
  }

  gpu.module @kernels {
    gpu.func @write(%in: memref<4xf32>, %out: memref<4xf32>) kernel {
      gpu.return
    }
  }
}
