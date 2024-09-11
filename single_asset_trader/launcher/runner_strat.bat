@echo off
SET "VENV_PATH=C:\Users\Admin\Documents\Pro\projets_code\python\trading_platform\.venv"
SET "SCRIPT_PATH=C:\Users\Admin\Documents\Pro\projets_code\python\trading_platform\single_asset_trader\launcher\runner_indicator_strat.py"

CALL "%VENV_PATH%\Scripts\activate.bat"
CALL "%VENV_PATH%\Scripts\python.exe" "%SCRIPT_PATH%" error_log.txt 2>&1 || pause
exit 0