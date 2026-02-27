
@echo off
echo Setting up clean virtual environment...
python -m venv venv
call venv\Scripts\activate
echo Installing core dependencies...
pip install sqlalchemy sqlalchemy[sqlite] exifread tqdm pillow numpy opencv-python-headless scikit-learn faiss-cpu fastapi uvicorn python-multipart
echo.
echo Setup Complete! 
echo To start the backend, run:
echo call venv\Scripts\activate
echo python main.py
pause
