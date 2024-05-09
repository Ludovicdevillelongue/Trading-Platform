@echo off
SET "VENV_PATH=C:\Users\ludov\Documents\pro\projets\trading_platform\venv"
SET "SCRIPT_PATH=C:\Users\ludov\Documents\pro\projets\trading_platform\launcher\runner_indicator_strat.py"

CALL "%VENV_PATH%\Scripts\activate.bat"
CALL "%VENV_PATH%\Scripts\python.exe" "%SCRIPT_PATH%" error_log.txt 2>&1 || pause
pause