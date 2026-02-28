@echo off
REM SmartGallery - Quick Start Script for Windows
REM This script helps you set up and run SmartGallery

echo ============================================================
echo    SmartGallery - Offline AI Photo Management System
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python is installed
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    pause
    exit /b 1
)

echo [OK] Node.js is installed
echo.

REM Main menu
:menu
echo ============================================================
echo                      MAIN MENU
echo ============================================================
echo.
echo 1. Install Dependencies (First Time Setup)
echo 2. Run Diagnostic Tests
echo 3. Build/Rebuild Index
echo 4. Start Backend Server
echo 5. Start Frontend Server
echo 6. Start Both (Backend + Frontend)
echo 7. Quick Verify
echo 8. Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto diagnostic
if "%choice%"=="3" goto build_index
if "%choice%"=="4" goto start_backend
if "%choice%"=="5" goto start_frontend
if "%choice%"=="6" goto start_both
if "%choice%"=="7" goto verify
if "%choice%"=="8" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:install
echo.
echo ============================================================
echo              Installing Dependencies
echo ============================================================
echo.

echo [1/3] Installing Python dependencies...
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    goto menu
)
cd ..

echo.
echo [2/3] Installing Node.js dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install Node.js dependencies
    pause
    goto menu
)
cd ..

echo.
echo [3/3] Checking Tesseract OCR...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Tesseract OCR is not installed
    echo Please install from: https://github.com/UB-Mannheim/tesseract/wiki
    echo OCR features will not work without it.
) else (
    echo [OK] Tesseract OCR is installed
)

echo.
echo ============================================================
echo [SUCCESS] Dependencies installed successfully!
echo ============================================================
echo.
echo Next steps:
echo 1. Add images to data/images/ folder
echo 2. Run option 3 to build index
echo 3. Run option 6 to start the application
echo.
pause
goto menu

:diagnostic
echo.
echo ============================================================
echo              Running Diagnostic Tests
echo ============================================================
echo.
cd backend
python comprehensive_diagnostic.py
cd ..
echo.
pause
goto menu

:build_index
echo.
echo ============================================================
echo              Building/Rebuilding Index
echo ============================================================
echo.

REM Check if images directory exists
if not exist "data\images" (
    echo [INFO] Creating data/images directory...
    mkdir data\images
)

REM Count images
set count=0
for %%f in (data\images\*.jpg data\images\*.jpeg data\images\*.png) do set /a count+=1

if %count%==0 (
    echo [WARNING] No images found in data/images/
    echo Please add some images first, then run this option again.
    echo.
    pause
    goto menu
)

echo [INFO] Found %count% images to process
echo.
echo This may take a few minutes depending on the number of images...
echo.

cd backend
python build_index.py
if errorlevel 1 (
    echo [ERROR] Index building failed
    pause
    goto menu
)
cd ..

echo.
echo ============================================================
echo [SUCCESS] Index built successfully!
echo ============================================================
echo.
pause
goto menu

:start_backend
echo.
echo ============================================================
echo              Starting Backend Server
echo ============================================================
echo.
echo Backend will start on http://localhost:8000
echo Press Ctrl+C to stop
echo.
cd backend
python main.py
cd ..
pause
goto menu

:start_frontend
echo.
echo ============================================================
echo              Starting Frontend Server
echo ============================================================
echo.
echo Frontend will start on http://localhost:3000
echo Press Ctrl+C to stop
echo.
cd frontend
call npm run dev
cd ..
pause
goto menu

:start_both
echo.
echo ============================================================
echo         Starting Backend and Frontend Servers
echo ============================================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Opening two command windows...
echo Close both windows to stop the servers.
echo.

REM Start backend in new window
start "SmartGallery Backend" cmd /k "cd backend && python main.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in new window
start "SmartGallery Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo [INFO] Servers are starting in separate windows...
echo [INFO] Wait a few seconds, then open http://localhost:3000
echo.
pause
goto menu

:verify
echo.
echo ============================================================
echo              Quick Verification
echo ============================================================
echo.
python quick_verify.py
echo.
pause
goto menu

:end
echo.
echo Thank you for using SmartGallery!
echo.
exit /b 0
