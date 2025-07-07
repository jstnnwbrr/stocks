# run_app.py
import subprocess
import os
import webbrowser
from threading import Timer

# --- Configuration ---
APP_FILE = "streamlit_app.py"
HOST = "localhost"
PORT = 8501

def open_browser():
    """
    Opens the default web browser to the Streamlit app's URL.
    """
    webbrowser.open(f"http://{HOST}:{PORT}")

if __name__ == "__main__":
    # Get the absolute path to the Streamlit app script
    # This is crucial for PyInstaller to find the file in the bundled app
    app_path = os.path.join(os.path.dirname(__file__), APP_FILE)

    # Command to run the Streamlit app
    command = [
        "streamlit", "run", app_path,
        "--server.headless", "true",  # Runs in headless mode, doesn't open a browser automatically
        "--server.port", str(PORT),
        "--server.address", HOST
    ]

    print(f"Starting Streamlit server with command: {' '.join(command)}")

    # Open the browser after a short delay to give the server time to start
    Timer(2, open_browser).start()

    # Start the Streamlit server process
    try:
        process = subprocess.Popen(command)
        process.wait()  # Wait for the process to complete (i.e., when the user closes the app/terminal)
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("'streamlit' command not found.")
        print("Please ensure Streamlit is installed and that its script directory is in your system's PATH.")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        input("Press Enter to exit...")