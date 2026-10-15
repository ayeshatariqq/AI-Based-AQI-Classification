# Aether — desktop app

A native Windows desktop application for single-reading AQI classification and batch CSV analysis. The interface uses the same warm off-white and muted green visual language as the Aether web app.

## Run on Windows

**Use Python 3.11 (recommended) or Python 3.12.** Python 3.14 may not have compatible prebuilt wheels for the pinned pandas/model dependencies.

1. Extract this folder.
2. Open PowerShell in this `aether_desktop` folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python app.py
   ```

If `py -3.11` is not found, install Python 3.11 from python.org and enable the Python Launcher during installation.

## Batch CSV analysis

Open **Batch analysis**, click **Open CSV**, or drag a CSV file onto the batch page. Use **Template** to save a correctly shaped example and **Export** to save predictions. CSV headers are matched without regard to case/outer spaces, UTF-8 BOMs are handled, and comma/semicolon-delimited files are detected.

Required columns: `City`, `Date` or `Month`, and all pollutant columns listed by the model. City names must be present in the model's supported city list; values must be numeric and non-negative. Invalid rows are marked **Skipped** with a reason instead of being silently classified.

Shortcuts: Ctrl+1..4 switch views · Ctrl+Enter analyze · Ctrl+O open CSV.

## Standalone executable

Run `build_exe.bat` from this folder after installing dependencies. Output is in `dist\Aether\`.

The app is an educational classification interface; predictions are model estimates, not official AQI readings or medical advice.


## Visual style
The desktop interface uses the same warm off-white background, forest-green accents, restrained borders, and sans-serif typography as the Aether web app.
