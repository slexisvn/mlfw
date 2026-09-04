@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
call "%VCVARS%" >nul || exit /b 1
if not exist "%TERAC%\build\build.ninja" (
  echo Not configured yet. Run scripts\configure.bat first.
  exit /b 1
)
cmake --build "%TERAC%/build" || exit /b 1
