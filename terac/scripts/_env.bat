@echo off
rem Every variable below can be preset in the environment. The defaults are
rem only used when it is not, so another checkout, another Visual Studio
rem edition or another ninja is a variable to set rather than a line to edit.

for %%i in ("%~dp0..") do set "TERAC=%%~fi"
for %%i in ("%TERAC%\..\..") do set "TERAC_SIBLING=%%~fi"

if not defined LLVM_SRC set "LLVM_SRC=%TERAC_SIBLING%\llvm-project"
if not defined LLVM_BUILD set "LLVM_BUILD=%LLVM_SRC%\build-assert"

if not defined NINJA for %%i in (ninja.exe) do if not "%%~$PATH:i"=="" set "NINJA=%%~$PATH:i"
if not defined NINJA set "NINJA=%TERAC_SIBLING%\ninja\ninja.exe"

if not defined VCVARS call :find_vcvars

rem CMake wants forward slashes on Windows; bench.bat turns them back where a
rem shell path is wanted.
set "LLVM_SRC=%LLVM_SRC:\=/%"
set "LLVM_BUILD=%LLVM_BUILD:\=/%"

if not exist "%VCVARS%" (
  echo error: no vcvars64.bat found. Set VCVARS to yours, e.g.
  echo   set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  exit /b 1
)
if not exist "%NINJA%" (
  echo error: ninja not found at "%NINJA%". Put it on PATH or set NINJA.
  exit /b 1
)
exit /b 0

:find_vcvars
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 0
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
exit /b 0
