#!/usr/bin/env python
"""
Launcher script for the Protify Streamlit GUI
"""
import subprocess
import sys
import os

def main():
    # Set environment variables to prevent PyTorch-Streamlit compatibility issues
    env = os.environ.copy()
    env["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    # Add src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Run streamlit with the modified environment
    streamlit_script = os.path.join('src', 'protify', 'streamlit_gui.py')
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', streamlit_script], env=env)

if __name__ == "__main__":
    main() 