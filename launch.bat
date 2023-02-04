@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0venv")

mkdir tmp 2>NUL

:check_python
%PYTHON% -c "" > tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 (echo Python not found. Please install Python 3.7 or higher. && :error_end)

:start_venv
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv
dir "%VENV_DIR%\Scripts\python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 (echo venv not found. Please run setup.bat before launching && goto :error_end)
goto :activate_venv

:activate_venv
call %VENV_DIR%\Scripts\activate.bat
set PYTHON=%VENV_DIR%\Scripts\python.exe
echo venv %PYTHON%

:skip_venv
jupyter-lab
