@echo off
rem Musubi Tuner GUI - start the local server and open the UI.
rem
rem Usage: start.bat [--browser] [--host HOST] [--port PORT]
rem   --browser   open the default web browser instead of a desktop window
rem   --host      interface to listen on (default 127.0.0.1)
rem   --port      port to listen on (default: MUSUBI_GUI_PORT, else 8000)
rem Every argument is passed through to "python -m app.launcher".
rem
rem Run install.bat once first. The trainer environment (..\musubi-tuner\.venv)
rem is put first on PATH so training jobs find its python and accelerate.
chcp 65001 >nul
setlocal EnableExtensions

set "B=%~dp0"
if "%B:~-1%"=="\" set "B=%B:~0,-1%"
for %%I in ("%B%\..") do set "ROOT=%%~fI"

if not exist "%B%\.venv\Scripts\python.exe" goto :noenv

if not defined MUSUBI_GUI_WORKSPACE_ROOT set "MUSUBI_GUI_WORKSPACE_ROOT=%ROOT%"

if exist "%ROOT%\musubi-tuner\.venv\Scripts\python.exe" goto :havetrainer
echo WARNING: the trainer environment was not found at "%ROOT%\musubi-tuner\.venv".
echo          Training needs it; run install.bat to set it up.
goto :run

:havetrainer
set "PATH=%ROOT%\musubi-tuner\.venv\Scripts;%PATH%"

:run
cd /d "%B%"
if errorlevel 1 goto :fail
rem Call the backend venv's python directly: "uv run" would put it first on PATH
rem and training jobs would pick it instead of the trainer environment.
"%B%\.venv\Scripts\python.exe" -m app.launcher %*
exit /b %ERRORLEVEL%

:noenv
echo ERROR: the backend environment "%B%\.venv" does not exist. Run install.bat first.
exit /b 1

:fail
echo ERROR: could not change to "%B%".
exit /b 1
