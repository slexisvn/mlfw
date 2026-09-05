// RUN: tera-opt %s --tera-stage-gpu-buffers --split-input-file | FileCheck %s

module attributes {gpu.container_module} {
  // A kernel operand is a host pointer and the device cannot read one, so each
  // buffer gets a device allocation beside it. Which copies that allocation
  // needs is read off the kernel: `@write` reads its first argument and writes
  // its second, so `%arg0` crosses on the way in and is freed as soon as the
  // launch that reads it is done, while `%alloc` crosses on the way back.
  //
  // `%alloc` is not copied in. It was allocated here and nothing has written
  // to it, so what the host holds is not worth the bandwidth.

  // CHECK-LABEL: func @two_kernels
  // CHECK-SAME:    (%[[ARG0:.*]]: memref<4xf32>)
  func.func @two_kernels(%arg0: memref<4xf32>) -> memref<4xf32> {
    // CHECK: %[[DEV0:.*]] = gpu.alloc
    %c1 = arith.constant 1 : index
    // CHECK: %[[HOST:.*]] = memref.alloc
    %alloc = memref.alloc() : memref<4xf32>
    // CHECK: %[[DEV1:.*]] = gpu.alloc
    // CHECK: gpu.memcpy %[[DEV0]], %[[ARG0]]
    // CHECK-NOT: gpu.memcpy %[[DEV1]], %[[HOST]]

    // CHECK: gpu.launch_func {{.*}} args(%[[DEV0]] : memref<4xf32>, %[[DEV1]] : memref<4xf32>)
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<4xf32>, %alloc : memref<4xf32>)

    // Nothing wrote `%arg0`, so it goes home without being read back.
    // CHECK-NOT: gpu.memcpy %[[ARG0]], %[[DEV0]]
    // CHECK: gpu.dealloc %[[DEV0]]

    // CHECK: gpu.launch_func {{.*}} args(%[[DEV1]] : memref<4xf32>, %[[DEV1]] : memref<4xf32>)
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<4xf32>, %alloc : memref<4xf32>)

    // CHECK: gpu.memcpy %[[HOST]], %[[DEV1]]
    // CHECK: gpu.dealloc %[[DEV1]]
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
  // The host reading a buffer between two kernels is a copy back and, if it
  // writes, a copy in again. It used to be an error, because a copy in placed
  // before every launch and a copy back placed after the last one leaves no
  // room for anything in between.
  //
  // Here the load only reads, so the device copy is still the current one when
  // the second kernel runs and nothing crosses on the way back in.

  // CHECK-LABEL: func @host_read_between_kernels
  func.func @host_read_between_kernels() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[HOST:.*]] = memref.alloc
    %alloc = memref.alloc() : memref<4xf32>
    // CHECK: %[[DEV:.*]] = gpu.alloc
    // CHECK: gpu.launch_func
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<4xf32>, %alloc : memref<4xf32>)
    // CHECK: gpu.memcpy %[[HOST]], %[[DEV]]
    // CHECK: memref.load %[[HOST]]
    %v = memref.load %alloc[%c0] : memref<4xf32>
    // CHECK-NOT: gpu.memcpy %[[DEV]], %[[HOST]]
    // CHECK: gpu.launch_func
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc : memref<4xf32>, %alloc : memref<4xf32>)
    // CHECK: gpu.dealloc %[[DEV]]
    memref.dealloc %alloc : memref<4xf32>
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

// -----

module attributes {gpu.container_module} {
  // A `tera.if` puts its launches inside a region, and the copies cannot follow
  // them in there: a copy back beside a launch in one branch runs only when that
  // branch is taken, and the allocation is then freed on one path and leaked on
  // the other. So both copies are placed against the ancestor of the launch that
  // stands in the buffer's own block -- here the `scf.if` itself.
  //
  // `%alloc` is copied in even though the kernel only writes it: a kernel that
  // writes some of a buffer leaves the rest as the host had it, and the store
  // above put something there.

  // CHECK-LABEL: func @launches_inside_a_branch
  // CHECK-SAME:    (%[[ARG0:.*]]: memref<4xf32>
  func.func @launches_inside_a_branch(%arg0: memref<4xf32>, %p: i1) -> memref<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[HOST:.*]] = memref.alloc
    %alloc = memref.alloc() : memref<4xf32>
    // CHECK: %[[V:.*]] = memref.load %[[ARG0]]
    %v = memref.load %arg0[%c0] : memref<4xf32>
    // CHECK: memref.store %[[V]], %[[HOST]]
    memref.store %v, %alloc[%c0] : memref<4xf32>

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

    // The copy back lands after the branch, where it runs whichever side was
    // taken, and each allocation is freed exactly once.
    // CHECK: }
    // CHECK: gpu.dealloc
    // CHECK: gpu.memcpy %[[HOST]]
    // CHECK: gpu.dealloc
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
    // CHECK: memref.copy %[[FLAT]], %[[VIEW]]
    // CHECK: gpu.dealloc %[[DEV]]
    // CHECK: memref.dealloc %[[FLAT]]
    return
  }

  gpu.module @kernels {
    gpu.func @write(%out: memref<4xf32, strided<[1], offset: ?>>) kernel {
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f32
      memref.store %zero, %out[%c0] : memref<4xf32, strided<[1], offset: ?>>
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
    gpu.func @write(%out: memref<?x2xf32>) kernel {
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f32
      memref.store %zero, %out[%c0, %c0] : memref<?x2xf32>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // A weight the caller uploaded once and means to keep there. The pointer is
  // already a device one, so nothing is allocated beside it and nothing
  // crosses; the batch next to it is staged as usual, which is what makes the
  // two kinds mixable in one call.
  //
  // The attribute is taken off once it has been honoured: below here the memref
  // is a pointer and the promise stops being checkable.

  // CHECK-LABEL: func @a_resident_weight
  // CHECK-SAME:    (%[[W:.*]]: memref<4xf32>, %[[X:.*]]: memref<4xf32>)
  // CHECK-NOT: tera.device_resident
  func.func @a_resident_weight(%w: memref<4xf32> {tera.device_resident},
                               %x: memref<4xf32>) {
    %c1 = arith.constant 1 : index
    // CHECK: %[[DEV:.*]] = gpu.alloc
    // CHECK: gpu.memcpy %[[DEV]], %[[X]]
    // CHECK: gpu.launch_func {{.*}} args(%[[W]] : memref<4xf32>, %[[DEV]] : memref<4xf32>)
    gpu.launch_func @kernels::@write blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%w : memref<4xf32>, %x : memref<4xf32>)
    // CHECK: gpu.memcpy %[[X]], %[[DEV]]
    // CHECK: gpu.dealloc %[[DEV]]
    return
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
