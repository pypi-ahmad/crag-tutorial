@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto missing_python
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))"
if errorlevel 1 goto missing_python
if exist ".venv\Scripts\python.exe" goto check_venv
set "VIRTUAL_ENV="
rem py -3 may prefer a free-threaded build lacking native dependency wheels.
rem A minor-version default selects its regular build, scoped by setlocal.
for /f "delims=" %%V in ('py -3 -c "import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))"') do set "PY_PYTHON3=%%V"
py -3 -c "import sys,sysconfig; sys.exit(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))"
if errorlevel 1 goto regular_python
py -3 -m venv .venv
if errorlevel 1 goto failed
:check_venv
".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info < (3, 11))"
if errorlevel 1 goto missing_python
".venv\Scripts\python.exe" -c "import sys,sysconfig; sys.exit(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))"
if errorlevel 1 goto regular_python
".venv\Scripts\python.exe" -m ensurepip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m ipykernel install --user --name crag-tutorial --display-name "Python (CRAG Tutorial)"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m notebook notebooks/01_crag_tutorial.ipynb
if errorlevel 1 goto failed
exit /b 0
:missing_python
echo Install Python 3.11 or newer with the Windows py launcher. Existing venv must also be 3.11+.
pause
exit /b 1
:regular_python
echo A regular, non-free-threaded Python 3.11+ build is required for native dependency wheels.
echo Install the regular build. If .venv is free-threaded, rename it as a backup before rerunning.
pause
exit /b 1
:failed
echo Setup or Jupyter failed. Read the error above, correct it, and rerun run.cmd.
pause
exit /b 1
