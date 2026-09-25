@echo off
cd /d "%~dp0"
python safetensor_inspector.py %*
if errorlevel 1 pause
