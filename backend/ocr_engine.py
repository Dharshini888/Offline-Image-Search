import pytesseract
from PIL import Image
import logging
import os

logger = logging.getLogger("OCREngine")

# Common Tesseract paths for Windows
TESS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Users\%USERNAME%\AppData\Local\Tesseract-OCR\tesseract.exe"
]

def _init_tesseract():
    try:
        pytesseract.get_tesseract_version()
    except:
        for p in TESS_PATHS:
            ep = os.path.expandvars(p)
            if os.path.exists(ep):
                pytesseract.pytesseract.tesseract_cmd = ep
                return
        logger.warning("Tesseract not found. OCR will be disabled.")

_init_tesseract()

def extract_text(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip() if text else ""
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return ""
