@echo off
setlocal

set "RELEASE_BASE_URL=https://github.com/microsoft/AirSim-Drone-Racing-Lab/releases/download/v1.0-windows"
set "ZIP_URL=%RELEASE_BASE_URL%/ADRL.zip"
set "SETTINGS_URL=%RELEASE_BASE_URL%/settings.json"

set "SCRIPT_DIR=%~dp0"
set "ZIP_PATH=%SCRIPT_DIR%ADRL.zip"
set "EXTRACT_DIR=%SCRIPT_DIR%"
set "ADRL_DIR=%SCRIPT_DIR%ADRL"
set "SETTINGS_PATH=%ADRL_DIR%\settings.json"

echo Downloading ADRL.zip...
where curl.exe >nul 2>nul
if %errorlevel%==0 (
    curl.exe -L -C - --retry 3 --retry-delay 2 -o "%ZIP_PATH%" "%ZIP_URL%"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile '%ZIP_PATH%'"
)
if errorlevel 1 goto :error

echo Extracting ADRL.zip...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '%EXTRACT_DIR%' -Force"
if errorlevel 1 goto :error

del /f /q "%ZIP_PATH%" >nul 2>nul

echo Downloading settings.json...
if not exist "%ADRL_DIR%" mkdir "%ADRL_DIR%"
where curl.exe >nul 2>nul
if %errorlevel%==0 (
    curl.exe -L --retry 3 --retry-delay 2 -o "%SETTINGS_PATH%" "%SETTINGS_URL%"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%SETTINGS_URL%' -OutFile '%SETTINGS_PATH%'"
)
if errorlevel 1 goto :error

echo Done.
exit /b 0

:error
echo Failed. Please check the error messages above.
exit /b 1
