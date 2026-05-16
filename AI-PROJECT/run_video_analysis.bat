@echo off
set "PYTHON_EXE=C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe"

cd /d "%~dp0"
"%PYTHON_EXE%" -m src.video_analyzer %*
pause
