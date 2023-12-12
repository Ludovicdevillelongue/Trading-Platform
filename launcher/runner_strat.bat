
@echo off
SETLOCAL ENABLEDELAYEDEXPANSION ENABLEEXTENSIONS

:: Settings
SET "script_path=C:\Users\ludov\Documents\pro\projets\trading_platform\launcher\runner_v1.py"
SET "python_path=C:\Users\ludov\anaconda3\python.exe"
SET "log_file=script_log.txt"
SET "virtual_env_path=C:\Users\ludov\anaconda3\envs\trading_platform"

:: Log start
echo [%date% - %time%] Starting script execution... > %log_file%

where %python_path% >nul 2>&1
IF !ERRORLEVEL! NEQ 0 (
    echo [%date% - %time%] ERROR: Python is not installed or not in PATH. >> %log_file%
    echo Python not found. Exiting...
    pause
    exit /b
)


:: Activate Virtual Environment (if needed)
IF EXIST "%virtual_env_path%\launcher\runner_strat.bat" (
    CALL "%virtual_env_path%\launcher\runner_strat.bat"
    echo [%date% - %time%] Activated virtual environment. >> %log_file%
) ELSE (
    echo [%date% - %time%] No virtual environment found. Using global Python environment. >> %log_file%
)

:: Optional: Setup additional environment variables
REM SET VAR_NAME=value

:: User Input (if required)
SET /P user_input=Enter value for script (Press Enter to skip):

:: Validate User Input (custom validation logic can be added here)
IF "!user_input!"=="" (
    echo [%date% - %time%] No user input provided. Proceeding with defaults... >> %log_file%
)

:: Run the Python script with user input and log output
echo [%date% - %time%] Running the Python script... >> %log_file%
%python_path% %script_path% !user_input! >> %log_file% 2>&1
SET "exit_code=!ERRORLEVEL!"

:: Check for script execution errors
IF !exit_code! NEQ 0 (
    echo [%date% - %time%] ERROR: Script exited with error code !exit_code!. >> %log_file%
    echo Error occurred. Check %log_file% for details.
    pause
    exit /b
)

:: Success message
echo [%date% - %time%] Script executed successfully. >> %log_file%
echo Script executed successfully. Check %log_file% for details.
pause
ENDLOCAL
