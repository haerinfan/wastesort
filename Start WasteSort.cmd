@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\launch.py
) else (
  py -3.12 scripts\launch.py
)
if errorlevel 1 (
  echo.
  echo WasteSort could not start. Read the message above.
  echo If Python was not found, install Python 3.12 from python.org.
  pause
)
