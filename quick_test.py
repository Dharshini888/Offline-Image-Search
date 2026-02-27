#!/usr/bin/env python3
import os, sys
os.chdir('backend')
sys.path.insert(0, '.')

# Test database only
from database import SessionLocal, init_db, Image as DBImage, Face as DBFace, Person
init_db()
db = SessionLocal()
print(f"DB: {db.query(DBImage).count()} images, {db.query(DBFace).count()} faces, {db.query(Person).count()} people")

# Test FAISS
import faiss
if os.path.exists("../data/index.faiss"):
    idx = faiss.read_index("../data/index.faiss")
    print(f"Image FAISS: {idx.ntotal} vectors")

if os.path.exists("../data/face_index.faiss"):
    fidx = faiss.read_index("../data/face_index.faiss")
    print(f"Face FAISS: {fidx.ntotal} vectors")

# Test basic imports
try:
    from search_engine import search_engine
    print("CLIP: OK (imported)")
except:
    print("CLIP: FAILED")

try:
    from detector_engine import detector_engine
    print(f"Detector: OK ({len(detector_engine.detector_engine.categories)} categories)")
except:
    print("Detector: FAILED")

try:
    from face_engine import face_engine
    print(f"Face Engine: {'OK' if face_engine.app else 'Not loaded'}")
except:
    print("Face Engine: FAILED")

try:
    from ocr_engine import extract_text
    print("OCR: OK (imported)")
except:
    print("OCR: FAILED")

db.close()
