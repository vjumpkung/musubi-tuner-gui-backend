@echo off
rem Musubi Tuner GUI - one-time Windows setup, then start the GUI.
rem
rem Usage: install.bat [--skip-trainer] [--rebuild-frontend] [--no-start]
rem   --skip-trainer       do not set up the musubi-tuner trainer environment
rem   --rebuild-frontend   rebuild the web UI even if it is already built
rem   --no-start           do not run start.bat when the install finishes
rem Environment: TORCH_INDEX_URL overrides the PyTorch wheel index
rem   (default https://download.pytorch.org/whl/cu128).
rem
rem This folder is the backend repository. Its parent folder is the workspace:
rem the web UI source goes to ..\musubi-tuner-gui-frontend and the trainer to
rem ..\musubi-tuner (with its own .venv). Requires git, and Node 20.19+ for the
rem web UI build. uv is installed automatically when it is missing.
chcp 65001 >nul
setlocal EnableExtensions

set "B=%~dp0"
if "%B:~-1%"=="\" set "B=%B:~0,-1%"
for %%I in ("%B%\..") do set "ROOT=%%~fI"

set "SKIP_TRAINER="
set "REBUILD_FRONTEND="
set "NO_START="
if not defined TORCH_INDEX_URL set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"

:parse
if "%~1"=="" goto :parsed
if /i "%~1"=="--skip-trainer" set "SKIP_TRAINER=1" & shift & goto :parse
if /i "%~1"=="--rebuild-frontend" set "REBUILD_FRONTEND=1" & shift & goto :parse
if /i "%~1"=="--no-start" set "NO_START=1" & shift & goto :parse
echo ERROR: unknown option "%~1". See the usage at the top of this file.
exit /b 1

:parsed
echo Workspace root: %ROOT%

rem --- 1. uv -------------------------------------------------------------
echo.
echo [1/5] Checking uv...
where uv >nul 2>nul
if not errorlevel 1 goto :have_uv
echo uv was not found; installing it...
powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.11.28/install.ps1 | iex"
if errorlevel 1 goto :fail
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 goto :fail_uv
:have_uv

rem --- 2. git ------------------------------------------------------------
echo.
echo [2/5] Checking git...
where git >nul 2>nul
if errorlevel 1 goto :fail_git

rem --- 3. backend environment -------------------------------------------
echo.
echo [3/5] Setting up the backend environment...
cd /d "%B%"
if errorlevel 1 goto :fail
call uv sync --python 3.12 --frozen --no-dev --extra desktop
if errorlevel 1 goto :fail

rem --- 4. web UI -----------------------------------------------------------
echo.
echo [4/5] Web UI...
if defined REBUILD_FRONTEND goto :build_frontend
if not exist "%B%\web\index.html" goto :build_frontend
echo Already built; use --rebuild-frontend to rebuild it.
goto :after_frontend

:build_frontend
where node >nul 2>nul
if errorlevel 1 goto :fail_node
node -e "const [a,b]=process.versions.node.split('.').map(Number);process.exit(a>20||(a===20&&b>=19)?0:1)"
if errorlevel 1 goto :fail_node
if exist "%ROOT%\musubi-tuner-gui-frontend" goto :have_frontend_src
git clone https://github.com/vjumpkung/musubi-tuner-gui-frontend.git "%ROOT%\musubi-tuner-gui-frontend"
if errorlevel 1 goto :fail
:have_frontend_src
cd /d "%ROOT%\musubi-tuner-gui-frontend"
if errorlevel 1 goto :fail
call npx --yes pnpm@10.24.0 install --frozen-lockfile
if errorlevel 1 goto :fail
call npx --yes pnpm@10.24.0 exec tsc
if errorlevel 1 goto :fail
call npx --yes pnpm@10.24.0 exec vite build --outDir "%B%\web" --emptyOutDir
if errorlevel 1 goto :fail
if not exist "%B%\web\index.html" goto :fail
cd /d "%B%"
if errorlevel 1 goto :fail
:after_frontend

rem --- 5. trainer ----------------------------------------------------------
echo.
echo [5/5] Trainer environment (musubi-tuner)...
if defined SKIP_TRAINER goto :skip_trainer
set "TRAINER_PY=%ROOT%\musubi-tuner\.venv\Scripts\python.exe"
if exist "%TRAINER_PY%" goto :trainer_present
if exist "%ROOT%\musubi-tuner" goto :have_trainer_src
git clone https://github.com/kohya-ss/musubi-tuner.git "%ROOT%\musubi-tuner"
if errorlevel 1 goto :fail
:have_trainer_src
call uv venv "%ROOT%\musubi-tuner\.venv" --python 3.12
if errorlevel 1 goto :fail
call uv pip install --python "%TRAINER_PY%" torch==2.11.0 torchvision==0.26.0 --index-url "%TORCH_INDEX_URL%"
if errorlevel 1 goto :fail
call uv pip install --python "%TRAINER_PY%" xformers==0.0.35 --index-url "%TORCH_INDEX_URL%"
if errorlevel 1 echo WARNING: xformers could not be installed; continuing without it.
call uv pip install --python "%TRAINER_PY%" -e "%ROOT%\musubi-tuner"
if errorlevel 1 goto :fail
goto :after_trainer

:trainer_present
echo Already installed at "%ROOT%\musubi-tuner\.venv".
goto :after_trainer

:skip_trainer
echo Skipped (--skip-trainer). Training needs it; run install.bat without that flag later.

:after_trainer
echo.
echo Installation finished.
echo   Workspace root: %ROOT%
echo   Start the GUI with: "%B%\start.bat"  (add --browser to use your web browser)
if defined NO_START exit /b 0
call "%B%\start.bat"
exit /b %ERRORLEVEL%

:fail_uv
echo ERROR: uv was installed but is not on PATH. Open a new terminal and run install.bat again.
goto :fail

:fail_git
echo ERROR: git is required. Install it from https://git-scm.com/download/win and run install.bat again.
goto :fail

:fail_node
echo ERROR: Node.js 20.19 or newer is required to build the web UI. Install it from https://nodejs.org/ and run install.bat again.
goto :fail

:fail
echo.
echo ERROR: installation failed; see the messages above.
exit /b 1
