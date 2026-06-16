# WHA Detector

This project is a Witch Hat Atelier symbol detector and uses a Python virtual environment with Homebrew Python and Tk support.

## Setup

1. Ensure Homebrew Python is installed.
2. Install Tk support for Python 3.14 if needed:
   ```bash
   brew install python-tk@3.14
   ```
3. Create and activate the virtual environment:
   ```bash
   /opt/homebrew/bin/python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

Use the venv Python to run the script:

```bash
./venv/bin/python draw.py
```

## Dependencies

- opencv-python
- Pillow
- numpy
- tkinter (provided by Homebrew `python-tk@3.14`)
