// RUN: tera-opt %s --split-input-file --verify-diagnostics --tera-stage-gpu-buffers

module attributes {gpu.container_module} {
  // A copy cannot go inside a region: it would run once per execution of the
  // region rather than once, and on a branch it would run only on the side that
  // was taken. So a region stands for everything inside it, and the copies go
  // around the whole thing.
  //
  // That works while the region is all kernel or all host. A region holding
  // both has an order between them that decides where the copy belongs, and
  // that order is not visible from out here, so it is reported rather than
  // guessed at.

  func.func @host_and_kernel_in_one_region(%p: i1) -> memref<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<4xf32>
    // expected-error @+1 {{holds both a kernel and a host access to the same buffer}}
    scf.if %p {
      gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<4xf32>, %alloc : memref<4xf32>)
      %v = memref.load %alloc[%c0] : memref<4xf32>
      memref.store %v, %alloc[%c1] : memref<4xf32>
    }
    return %alloc : memref<4xf32>
  }

  gpu.module @kernels {
    gpu.func @write(%in: memref<4xf32>, %out: memref<4xf32>) kernel {
      %c0 = arith.constant 0 : index
      %v = memref.load %in[%c0] : memref<4xf32>
      memref.store %v, %out[%c0] : memref<4xf32>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // Staging a buffer and a view of it separately would put the same elements
  // on the device twice, and each copy back would overwrite what the other one
  // brought. The two would have to share an allocation, which is a question
  // about overlap rather than about aliasing, so it is refused instead.

  func.func @whole_and_view_both_staged(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    // expected-error @+1 {{and so is the buffer it views}}
    %view = memref.subview %arg0[%c1] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
    gpu.launch_func @kernels::@whole blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<8xf32>)
    gpu.launch_func @kernels::@part blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%view : memref<4xf32, strided<[1], offset: ?>>)
    return
  }

  gpu.module @kernels {
    gpu.func @whole(%out: memref<8xf32>) kernel {
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f32
      memref.store %zero, %out[%c0] : memref<8xf32>
      gpu.return
    }
    gpu.func @part(%out: memref<4xf32, strided<[1], offset: ?>>) kernel {
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f32
      memref.store %zero, %out[%c0] : memref<4xf32, strided<[1], offset: ?>>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // The pointer behind a `tera.device_resident` argument is one the host cannot
  // dereference, so a host access to it is not a copy that could be inserted --
  // it is a fault waiting for the first read.

  func.func @host_reads_a_resident_weight(%w: memref<4xf32> {tera.device_resident})
      -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%w : memref<4xf32>, %w : memref<4xf32>)
    // expected-error @+1 {{reads a buffer the caller left on the device}}
    %v = memref.load %w[%c0] : memref<4xf32>
    return %v : f32
  }

  gpu.module @kernels {
    gpu.func @write(%in: memref<4xf32>, %out: memref<4xf32>) kernel {
      %c0 = arith.constant 0 : index
      %v = memref.load %in[%c0] : memref<4xf32>
      memref.store %v, %out[%c0] : memref<4xf32>
      gpu.return
    }
  }
}
