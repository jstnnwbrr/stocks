@echo off
REM Path to your virtual environment's activation script
call "C:\Users\sar81\fortune-teller-env\Scripts\activate.bat"

REM Path to your forecasting script
python "C:\Users\sar81\stocks\run_app.py"

pause
