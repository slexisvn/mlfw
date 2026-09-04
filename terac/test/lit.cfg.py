# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

config.name = "TERA"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.tera_obj_root, "test")

# The 'Inputs' subdirectories hold auxiliary data for the tests beside them,
# not tests of their own.
config.excludes = ["Inputs"]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

config.tera_tools_dir = os.path.join(config.tera_obj_root, "bin")
llvm_config.add_tool_substitutions(
    ["tera-capi-test", "tera-gradcheck", "tera-opt", "tera-runner"],
    [config.tera_tools_dir, config.llvm_tools_dir],
)

# The differential harness runs the mlfw compiler for ground truth. It needs
# node and a built mlfw, neither of which this project owns, so the tests that
# use it are gated on the feature rather than assumed.
config.tera_harness_dir = os.path.join(config.tera_source_root, "harness")
config.substitutions.append(("%tera_harness", config.tera_harness_dir))

# A lowered memref program can call into MLIR's C runtime helpers, so the JIT
# has to be handed the library holding them.
for directory in [
    config.llvm_tools_dir,
    os.path.join(os.path.dirname(config.llvm_tools_dir), "lib"),
]:
    for stem in ["mlir_c_runner_utils", "libmlir_c_runner_utils"]:
        candidate = os.path.join(directory, stem + config.llvm_shlib_ext)
        if os.path.exists(candidate):
            config.substitutions.append(("%mlir_c_runner_utils", candidate))
            break
    else:
        continue
    break

# Running on the device needs the CUDA runtime wrappers to link the kernel
# launches against, and a device to launch them on. The library is built only
# when LLVM was configured with MLIR_ENABLE_CUDA_RUNNER, and `nvidia-smi` is
# the cheapest thing that answers whether a driver and a card are there at all.
for directory in [
    config.llvm_tools_dir,
    os.path.join(os.path.dirname(config.llvm_tools_dir), "lib"),
]:
    for stem in ["mlir_cuda_runtime", "libmlir_cuda_runtime"]:
        candidate = os.path.join(directory, stem + config.llvm_shlib_ext)
        if os.path.exists(candidate):
            config.substitutions.append(("%mlir_cuda_runtime", candidate))
            if lit.util.which("nvidia-smi"):
                config.available_features.add("cuda")
            break
    else:
        continue
    break

node = lit.util.which("node")
mlfw_bundle = os.path.join(
    os.path.dirname(config.tera_source_root), "dist", "index.node.js"
)
if node and os.path.exists(mlfw_bundle):
    config.available_features.add("mlfw-oracle")
    # Quoted: on Windows node usually lives under a path with a space in it.
    config.substitutions.append(("%node", '"' + node + '"'))
