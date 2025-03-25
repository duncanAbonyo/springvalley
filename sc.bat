@echo off
title Python Script Auto-Restarter

:: Define the script name (assumes it's in the same directory)
set SCRIPT_NAME=app.py

:: Loop indefinitely
:restart
echo Starting script...
start /B python %SCRIPT_NAME%

:: Wait for 15 minutes (900 seconds)
timeout /t 900 /nobreak

:: Kill Python processes
echo Stopping script...
taskkill /IM python.exe /F

:: Restart the script
goto restart
