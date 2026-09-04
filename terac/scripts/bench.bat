@echo off
setlocal enabledelayedexpansion
call "%~dp0_env.bat" || exit /b 1

set "RUNS=%~1"
if "%RUNS%"=="" set "RUNS=20"

set "RUNNER=%TERAC%\build\bin\tera-runner.exe"
set "OPT=%TERAC%\build\bin\tera-opt.exe"
set "LIBS=%LLVM_BUILD%/bin/mlir_c_runner_utils.dll"
rem The runtime helpers pull in LLVM's own DLLs, and a bare shell does
rem not have them on PATH. Loading them without this fails as 0x7E, which
rem names the missing dependency rather than the library that needed it.
set "PATH=%LLVM_BUILD:/=\%\bin;%PATH%"
if not exist "%RUNNER%" (
  echo Not built yet. Run scripts\build.bat first.
  exit /b 1
)

for %%M in (mlp attention rnn) do (
  set "MODEL=%TERAC%\test\bench\%%M.mlir"
  "%RUNNER%" "!MODEL!" --entry=%%M --benchmark=%RUNS% --shared-libs=%LIBS% || exit /b 1
  "%OPT%" "!MODEL!" --tera-autodiff | "%RUNNER%" - --entry=%%M_vjp --benchmark=%RUNS% --shared-libs=%LIBS% || exit /b 1
)
