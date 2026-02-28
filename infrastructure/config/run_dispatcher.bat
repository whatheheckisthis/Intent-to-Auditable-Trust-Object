@echo off
setlocal EnableDelayedExpansion

set "ENV_FILE=.env"
set "LOG_FILE=dispatcher-error.log"

if exist "%ENV_FILE%" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    if not "%%~A"=="" (
      if /I not "%%~A:~0,1"=="#" set "%%~A=%%~B"
    )
  )
)

if "%API_KEY%"=="" (
  echo ERROR: API_KEY not found in environment or .env
  exit /b 10
)

if "%REST_URL%"=="" (
  echo ERROR: REST_URL not found in environment or .env
  exit /b 11
)

if "%~1"=="" (
  echo Usage: run_dispatcher.bat ^<policy.json^>
  exit /b 12
)

dispatcher.exe "%~1"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo [%DATE% %TIME%] dispatcher failed with code %RC% >> "%LOG_FILE%"
  exit /b %RC%
)

exit /b 0
