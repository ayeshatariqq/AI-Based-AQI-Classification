@echo off
REM Builds a standalone Windows app into dist\Aether\  (no Python needed to run it)
pip install pyinstaller
pyinstaller --noconfirm --windowed --name Aether ^
  --add-data "models;models" --add-data "fonts;fonts" ^
  --collect-all xgboost --collect-all imblearn ^
  app.py
echo.
echo Done. Run dist\Aether\Aether.exe
