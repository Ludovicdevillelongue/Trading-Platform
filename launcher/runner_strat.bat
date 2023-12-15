@echo off
SET "CONDA_PATH=C:\Users\ludov\Anaconda3"
SET "ENV_NAME=trading_platform"
SET "SCRIPT_PATH=C:\Users\ludov\Documents\pro\projets\trading_platform\launcher\runner_indicator_strat.py"

CALL "%CONDA_PATH%\Scripts\activate.bat" %CONDA_PATH%
CALL conda activate %ENV_NAME%
CALL python "%SCRIPT_PATH%"
pause