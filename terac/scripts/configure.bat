@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
call "%VCVARS%" >nul || exit /b 1
if not exist "%LLVM_BUILD%/lib/cmake/mlir" (
  echo LLVM_BUILD not found: %LLVM_BUILD%
  echo Run scripts\build-llvm-assert.bat first, or set LLVM_BUILD to your build.
  exit /b 1
)
if exist "%TERAC%\build\CMakeCache.txt" (
  findstr /C:"CMAKE_GENERATOR:INTERNAL=Ninja" "%TERAC%\build\CMakeCache.txt" >nul || (
    echo build\ was configured with a different generator. CMake will not switch in place.
    echo Delete it first:  rmdir /s /q "%TERAC%\build"
    exit /b 1
  )
)
cmake -G Ninja ^
  -S "%TERAC%" ^
  -B "%TERAC%/build" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DMLIR_DIR=%LLVM_BUILD%/lib/cmake/mlir ^
  -DMLIR_INCLUDE_TESTS=ON ^
  -DLLVM_EXTERNAL_LIT=%LLVM_BUILD%/bin/llvm-lit.py ^
  -DCMAKE_MAKE_PROGRAM=%NINJA% || exit /b 1
