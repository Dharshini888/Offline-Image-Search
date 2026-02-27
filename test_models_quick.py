#!/usr/bin/env python3
"""Quick test of all models (without search endpoint)"""
import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, '.')

print("=" * 60)
print("🔍 MODEL HEALTH CHECK")
print("=" * 60)

# 1. DATABASE
print("\n1️⃣  DATABASE")
try:
    from database import SessionLocal, init_db, Image as DBImage
    init_db()
    db = SessionLocal()
    img_count = db.query(DBImage).count()
    print(f"   ✅ Database OK ({img_count} images)")
    db.close()
except Exception as e:
    print(f"   ❌ Database FAILED: {e}")

# 2. FAISS INDICES
print("\n2️⃣  FAISS INDICES")
try:
    import faiss
    if os.path.exists("../data/index.faiss"):
        idx = faiss.read_index("../data/index.faiss")
        print(f"   ✅ Image index OK ({idx.ntotal} vectors)")
    else:
        print(f"   ❌ Image index missing")
    
    if os.path.exists("../data/face_index.faiss"):
        fidx = faiss.read_index("../data/face_index.faiss")
        print(f"   ✅ Face index OK ({fidx.ntotal} vectors)")
    else:
        print(f"   ❌ Face index missing")
except Exception as e:
    print(f"   ❌ FAISS FAILED: {e}")

# 3. CLIP MODEL
print("\n3️⃣  CLIP MODEL")
try:
    from search_engine import search_engine
    emb = search_engine.get_text_embedding("test")
    if emb is not None:
        print(f"   ✅ CLIP OK (embedding dim: {emb.shape[0]})")
    else:
        print(f"   ❌ CLIP FAILED")
except Exception as e:
    print(f"   ❌ CLIP ERROR: {e}")

# 4. FACE RECOGNITION
print("\n4️⃣  FACE RECOGNITION")
try:
    from face_engine import face_engine
    if face_engine.app:
        print(f"   ✅ InsightFace OK")
    else:
        print(f"   ❌ InsightFace not loaded")
except Exception as e:
    print(f"   ❌ Face engine ERROR: {e}")

# 5. OCR
print("\n5️⃣  OCR MODEL")
try:
    from ocr_engine import extract_text
    print(f"   ✅ OCR available")
except Exception as e:
    print(f"   ❌ OCR ERROR: {e}")

# 6. OBJECT DETECTION
print("\n6️⃣  OBJECT DETECTION")
try:
    from detector_engine import detector_engine
    cats = detector_engine.detector_engine.categories
    print(f"   ✅ Detector OK ({len(cats)} categories)")
    print(f"      Categories: {cats[:5]}")
except Exception as e:
    print(f"   ❌ Detector ERROR: {e}")

print("\n" + "=" * 60)
print("✅ HEALTH CHECK COMPLETE")
print("=" * 60)
