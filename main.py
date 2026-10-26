"""Command-line entry point for the Streamlit application."""

import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
