@echo off
REM Batch file to run Protify Streamlit GUI on Windows
REM This sets the necessary environment variables to prevent PyTorch-Streamlit compatibility issues

echo Starting Protify Streamlit GUI...

REM Set environment variables to prevent PyTorch-Streamlit compatibility issues
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

REM Change to the src/protify directory and run Streamlit
cd src\protify
py -m streamlit run streamlit_gui.py

pause