@echo off
cd /d "%~dp0"

echo.
echo  ================================
echo    E-PROCTORING SYSTEM
echo  ================================
echo   1.  Webcam Mode
echo   2.  Video Analysis Mode
echo  ================================
echo.
set /p CHOICE="  Choose [1/2]: "

if "%CHOICE%"=="1" goto webcam
if "%CHOICE%"=="2" goto video
echo  [ERROR] Invalid choice. Please enter 1 or 2.
goto end

:webcam
echo.
python main.py
goto end

:video
echo.
set /p VIDEO_PATH="  Video file path: "
set /p LABELS_PATH="  Labels segment CSV path: "
set /p STUDENT_ID="  Student ID (must match labels_segment.csv): "
echo.
if "%LABELS_PATH%"=="" (
    python main.py --video "%VIDEO_PATH%" --student "%STUDENT_ID%"
) else (
    python main.py --video "%VIDEO_PATH%" --labels "%LABELS_PATH%" --student "%STUDENT_ID%"
)
goto end

:end
echo.
pause
