// RUN: tera-opt %s --tera-stage-gpu-buffers --split-input-file | FileCheck %s

module attributes {gpu.container_module} {
  // A kernel operand is a host pointer and the device cannot read one, so each
  // buffer gets a device allocation beside it, filled just before the first
  // launch that uses it and read back after the last one. Last use is per buffer
  // and not per function: `%arg0` is done with after the first kernel, `%alloc`
  // is not.
  //
  // The copy in sits before the first launch rather than beside the allocation
  // so that whatever the host wrote on its way there is what crosses.

  // CHECK-LABEL: func @two_kernels
  // CHECK-SAME:    (%[[ARG0:.*]]: memref<4xf32>)
  func.func @two_kernels(%arg0: memref<4xf32>) -> memref<4xf32> {
    // CHECK: %[[DEV0:.*]] = gpu.alloc
    %c1 = arith.constant 1 : index
    // CHECK: %[[HOST:.*]] = memref.alloc
    %alloc = memref.alloc() : memref<4xf32>
    // CHECK: %[[DEV1:.*]] = gpu.alloc
    // CHECK: gpu.memcpy %[[DEV0]], %[[ARG0]]
    // CHECK: gpu.memcpy %[[DEV1]], %[[HOST]]

    // CHECK: gpu.launch_func {{.*}} args(%[[DEV0]] : memref<4xf32>, %[[DEV1]] : memref<4xf32>)
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<4xf32>, %alloc : memref<4xf32>)

    // CHECK: gpu.memcpy %[[ARG0]], %[[DEV0]]
    // CHECK: gpu.dealloc %[[DEV0]]

    // CHECK: gpu.launch_func {{.*}} args(%[[DEV1]] : memref<4xf32>, %[[DEV1]] : memref<4xf32>)
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<4xf32>, %alloc : memref<4xf32>)

    // CHECK: gpu.memcpy %[[HOST]], %[[DEV1]]
    // CHECK: gpu.dealloc %[[DEV1]]
    return %alloc : memref<4xf32>
  }

  gpu.module @kernels {
    gpu.func @write(%in: memref<4xf32>, %out: memref<4xf32>) kernel {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // A `tera.if` puts its launches inside a region, and the copies cannot follow
  // them in there: a copy back beside a launch in one branch runs only when that
  // branch is taken, and the allocation is then freed on one path and leaked on
  // the other. So both copies are placed against the ancestor of the launch that
  // stands in the buffer's own block -- here the `scf.if` itself.
  //
  // The host load is the second half of it. It reads `%arg0` before any kernel
  // runs, which is safe, but it cannot be ordered against a launch nested one
  // region down; a check that could only look downwards used to reject it.

  // CHECK-LABEL: func @launches_inside_a_branch
  // CHECK-SAME:    (%[[ARG0:.*]]: memref<4xf32>
  func.func @launches_inside_a_branch(%arg0: memref<4xf32>, %p: i1) -> memref<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[HOST:.*]] = memref.alloc
    %alloc = memref.alloc() : memref<4xf32>
    // CHECK: memref.load %[[ARG0]]
    %v = memref.load %arg0[%c0] : memref<4xf32>

    // CHECK: gpu.memcpy
    // CHECK: gpu.memcpy
    // CHECK: scf.if
    scf.if %p {
      // CHECK-NOT: gpu.memcpy
      gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<4xf32>, %alloc : memref<4xf32>)
    } else {
      // CHECK-NOT: gpu.memcpy
      gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<4xf32>, %alloc : memref<4xf32>)
    }

    // Both copies back land after the branch, where they run whichever side was
    // taken, and each allocation is freed exactly once.
    // CHECK: }
    // CHECK: gpu.memcpy %[[HOST]]
    // CHECK: gpu.dealloc
    // CHECK: gpu.memcpy %[[ARG0]]
    // CHECK: gpu.dealloc
    return %alloc : memref<4xf32>
  }

  gpu.module @kernels {
    gpu.func @write(%in: memref<4xf32>, %out: memref<4xf32>) kernel {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // A `tera.scan` step slice reaches a launch as a memref with a runtime offset,
  // and `gpu.memcpy` copies whole buffers rather than walking strides. So the
  // view is copied into a contiguous buffer on the host, that buffer is what
  // crosses, and it is copied back into the view afterwards.
  //
  // The launch operand is cast back to the type the outlined kernel declares
  // rather than the kernel's signature being rewritten: the buffer behind the
  // cast is contiguous, so the offset and strides the cast makes dynamic are
  // exactly the ones the kernel body was already computing with.

  // CHECK-LABEL: func @strided_operand
  func.func @strided_operand(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    // CHECK: %[[VIEW:.*]] = memref.subview
    %view = memref.subview %arg0[%c1] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
    // CHECK: %[[FLAT:.*]] = memref.alloc() : memref<4xf32>
    // CHECK: %[[DEV:.*]] = gpu.alloc
    // CHECK: memref.copy %[[VIEW]], %[[FLAT]]
    // CHECK: gpu.memcpy %[[DEV]], %[[FLAT]]
    // CHECK: %[[CAST:.*]] = memref.cast %[[DEV]] : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
    // CHECK: gpu.launch_func {{.*}} args(%[[CAST]] : memref<4xf32, strided<[1], offset: ?>>)
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%view : memref<4xf32, strided<[1], offset: ?>>)
    // CHECK: gpu.memcpy %[[FLAT]], %[[DEV]]
    // CHECK: gpu.dealloc %[[DEV]]
    // CHECK: memref.copy %[[FLAT]], %[[VIEW]]
    // CHECK: memref.dealloc %[[FLAT]]
    return
  }

  gpu.module @kernels {
    gpu.func @write(%in: memref<4xf32, strided<[1], offset: ?>>) kernel {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // A dynamic extent is a number the descriptor is already carrying, so both
  // allocations are told it with `memref.dim` rather than the buffer being
  // refused. Without this a program with a `?` in it lowered, ran, and gave the
  // right answer entirely on the host: the kernels were outlined and then the
  // staging pass turned them back into an error.

  // CHECK-LABEL: func @dynamic_operand
  // CHECK-SAME:    (%[[ARG0:.*]]: memref<?x2xf32>)
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[N:.*]] = memref.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[DEV:.*]] = gpu.alloc {{.*}}(%[[N]])
  func.func @dynamic_operand(%arg0: memref<?x2xf32>) {
    %c1 = arith.constant 1 : index
    // CHECK: gpu.memcpy %[[DEV]], %[[ARG0]]
    // CHECK: gpu.launch_func {{.*}} args(%[[DEV]] : memref<?x2xf32>)
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<?x2xf32>)
    // CHECK: gpu.memcpy %[[ARG0]], %[[DEV]]
    // CHECK: gpu.dealloc %[[DEV]]
    return
  }

  gpu.module @kernels {
    gpu.func @write(%in: memref<?x2xf32>) kernel {
      gpu.return
    }
  }
}
