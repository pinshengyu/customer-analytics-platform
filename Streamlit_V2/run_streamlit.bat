@echo off
cd /d "%~dp0"
python ..\launch_site.py
if errorlevel 1 pause
