@echo off
cd /d "%~dp0"

python -m src.video_analyzer %*
pause
