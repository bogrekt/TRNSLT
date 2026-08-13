@echo off
rem Запуск trnslt без чёрного окна консоли.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app.py"
