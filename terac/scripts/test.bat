@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
call "%VCVARS%" >nul || exit /b 1
cmake --build "%TERAC%/build" --target check-tera || exit /b 1
