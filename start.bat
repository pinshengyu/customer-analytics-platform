@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
python launch_site.py
if errorlevel 1 pause
