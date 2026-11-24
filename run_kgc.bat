@echo off
setlocal
title KGC_main
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KGC_EXIT_CODE=1"
pushd "%~dp0"

if defined CONDA_EXE if exist "%CONDA_EXE%" goto configured_conda
where conda >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%P in ('where conda') do (
        set "CONDA_EXE=%%P"
        goto configured_conda
    )
)
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
    if exist "%%~P\Scripts\conda.exe" (
        set "CONDA_EXE=%%~P\Scripts\conda.exe"
        goto configured_conda
    )
)
if exist "%USERPROFILE%\.conda\environments.txt" (
    for /f "usebackq delims=" %%P in ("%USERPROFILE%\.conda\environments.txt") do (
        if exist "%%P\Scripts\conda.exe" (
            set "CONDA_EXE=%%P\Scripts\conda.exe"
            goto configured_conda
        )
    )
)
echo Conda was not found. Open Anaconda Prompt and run:
echo   conda activate KGC
echo   python "%~dp0run.py"
goto finished

:configured_conda
for %%P in ("%CONDA_EXE%") do set "CONDA_LAUNCHER_DIR=%%~dpP"
if exist "%CONDA_LAUNCHER_DIR%..\python.exe" (
    set "CONDA_EXE=%CONDA_LAUNCHER_DIR%..\python.exe"
    set "_CE_M=-m"
    set "_CE_CONDA=conda"
)
call "%CONDA_EXE%" %_CE_M% %_CE_CONDA% run --no-capture-output -n KGC python -u "%~dp0run.py" %*
set "KGC_EXIT_CODE=%ERRORLEVEL%"

:finished
echo.
pause
popd
endlocal & exit /b %KGC_EXIT_CODE%
