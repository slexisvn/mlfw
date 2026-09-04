@echo off
rem Usage: build-llvm-assert.bat [jobs]   default 14, leaves headroom on a 20-core / 32 GB box.
rem
rem MLIR cannot be built without LLVM -- it is a project on top of it, and the MLIR
rem libraries link LLVMSupport/LLVMCore, plus the X86 backend for the ExecutionEngine.
rem What CAN be skipped is the 87 llvm-* command line tools (llc, opt, llvm-ar, ...),
rem which an out-of-tree dialect never uses. LLVM_BUILD_TOOLS=OFF drops them from the
rem default target; `ninja llc` still builds one by name if you ever need it.
rem LLVM_BUILD_UTILS stays ON -- FileCheck, count and not live there, and lit needs them.
setlocal
call "%~dp0_env.bat" || exit /b 1
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=14"
call "%VCVARS%" >nul || exit /b 1
cmake -G Ninja ^
  -S %LLVM_SRC%/llvm ^
  -B %LLVM_BUILD% ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_ENABLE_ASSERTIONS=ON ^
  -DLLVM_ENABLE_PROJECTS=mlir ^
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" ^
  -DLLVM_INCLUDE_TESTS=ON ^
  -DLLVM_OPTIMIZED_TABLEGEN=ON ^
  -DLLVM_BUILD_TOOLS=OFF ^
  -DLLVM_BUILD_UTILS=ON ^
  -DLLVM_INCLUDE_EXAMPLES=OFF ^
  -DLLVM_INCLUDE_BENCHMARKS=OFF ^
  -DLLVM_INCLUDE_DOCS=OFF ^
  -DLLVM_PARALLEL_COMPILE_JOBS=%JOBS% ^
  -DLLVM_PARALLEL_LINK_JOBS=4 ^
  -DCMAKE_MAKE_PROGRAM=%NINJA% || exit /b 1
cmake --build %LLVM_BUILD% -- -j %JOBS% || exit /b 1
echo.
echo === LLVM assertions build done (jobs=%JOBS%) ===
