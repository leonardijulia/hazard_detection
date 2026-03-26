@echo off
echo Starting Prithvi Hazard Detection Backend...
:: This uses the python inside your venv to run uvicorn
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
pause