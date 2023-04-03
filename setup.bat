@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0venv")
if not defined GIT (set GIT=git)

mkdir tmp 2>NUL

:check_python
%PYTHON% -c "" > tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 (echo Python not found. Please install Python 3.7 or higher. && goto :error_end)

:check_git
%GIT% --version > tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 (echo Git not found. Please install Git. && goto :error_end)


for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print (sys.executable)"') do set "PYTHON_FULL_PATH=%%i"
echo Using Python executable: %PYTHON_FULL_PATH%
echo Using venv directory: %VENV_DIR%


:start_venv
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv
dir "%VENV_DIR%\Scripts\python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 (goto :create_venv)
goto :activate_venv

:create_venv
echo Creating virtual environment...
%PYTHON_FULL_PATH% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 (echo Failed to create virtual environment. && goto :error_end)
goto :activate_venv

:activate_venv
call %VENV_DIR%\Scripts\activate.bat
set PYTHON=%VENV_DIR%\Scripts\python.exe
echo venv %PYTHON%


:skip_venv


:install_requirements
echo Installing requirements...

echo Cloning v-diffusion-pytorch repository
%GIT% clone --recursive https://github.com/crowsonkb/v-diffusion-pytorch
if %ERRORLEVEL% NEQ 0 (echo Failed to clone v-diffusion-pytorch repository. && goto :error_end)

echo Installing sample-generator
%PYTHON% -m pip install -e .
if %ERRORLEVEL% NEQ 0 (echo Failed to install sample-generator && goto :error_end)

echo Installing v-diffusion-pytorch
%PYTHON% -m pip install v-diffusion-pytorch
if %ERRORLEVEL% NEQ 0 (echo Failed to install v-diffusion-pytorch && goto :error_end)

echo Installing ipywidgets==7.7.2
%PYTHON% -m pip install ipywidgets==7.7.2
if %ERRORLEVEL% NEQ 0 (echo Failed to install ipywidgets==7.7.2 && goto :error_end)

echo Installing k-diffusion
%PYTHON% -m pip install k-diffusion
if %ERRORLEVEL% NEQ 0 (echo Failed to install k-diffusion && goto :error_end)

echo Installing matplotlib
%PYTHON% -m pip install matplotlib
if %ERRORLEVEL% NEQ 0 (echo Failed to install matplotlib && goto :error_end)

echo Installing soundfile
%PYTHON% -m pip install soundfile
if %ERRORLEVEL% NEQ 0 (echo Failed to install soundfile && goto :error_end)

echo Installing jupyterlab
%PYTHON% -m pip install jupyterlab
if %ERRORLEVEL% NEQ 0 (echo Failed to install jupyterlab && goto :error_end)

echo Installing pytorch and CUDA
%PYTHON% -m pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
if %ERRORLEVEL% NEQ 0 (echo Failed to install pytorch && goto :error_end)


echo Setup complete! You can now run launch.bat to launch jupyter-lab in the virtual environment.
pause

goto :end


goto :end
:error_end
echo An error occurred. Please see the output above for more information.
pause


:end
