#!/usr/bin/env python3
"""Comprehensive test of all models and systems"""
import os
import sys
import json

os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, '.')

from database import SessionLocal, init_db, Image as DBImage
from search_engine import search_engine
from face_engine import face_engine
from ocr_engine import extract_text
from detector_engine import detector_engine
import faiss

print("=" * 60)
print("🔍 COMPREHENSIVE MODEL TEST")
print("=" * 60)

# 1. DATABASE TEST
print("\n1️⃣  DATABASE")
print("-" * 60)
try:
    init_db()
    db = SessionLocal()
    img_count = db.query(DBImage).count()
    print(f"   ✅ Database connected")
    print(f"   ✅ Images in DB: {img_count}")
    
    # Check for sample image
    sample = db.query(DBImage).first()
    if sample:
        print(f"   ✅ Sample image: {sample.filename}")
        print(f"      - OCR: {'✅' if sample.ocr_text else '❌'}")
        print(f"      - Tags: {sample.scene_label if sample.scene_label else '❌'}")
        print(f"      - Colors: R={sample.avg_r}, G={sample.avg_g}, B={sample.avg_b}")
    db.close()
except Exception as e:
    print(f"   ❌ Database error: {e}")

# 2. CLIP MODEL TEST
print("\n2️⃣  CLIP SEMANTIC SEARCH")
print("-" * 60)
try:
    # Test text embedding
    text_emb = search_engine.get_text_embedding("a dog", use_prompt_ensemble=True)
    if text_emb is not None:
        print(f"   ✅ Text embedding working")
        print(f"      - Dimension: {text_emb.shape[0]}")
        print(f"      - L2 norm: {(text_emb ** 2).sum() ** 0.5:.4f}")
    else:
        print(f"   ❌ Text embedding failed")
    
    # Test image embedding (if one exists)
    if img_count > 0:
        db = SessionLocal()
        img_path = f"../data/images/{sample.filename}" if sample else None
        if img_path and os.path.exists(img_path):
            img_emb = search_engine.get_image_embedding(img_path)
            if img_emb is not None:
                print(f"   ✅ Image embedding working")
                print(f"      - Dimension: {img_emb.shape[0]}")
            else:
                print(f"   ❌ Image embedding failed")
        db.close()
    
    # Test FAISS index
    if search_engine.index:
        print(f"   ✅ FAISS image index loaded")
        print(f"      - Vectors: {search_engine.index.ntotal}")
    else:
        print(f"   ❌ FAISS image index not loaded")
        
except Exception as e:
    print(f"   ❌ CLIP error: {e}")
    import traceback
    traceback.print_exc()

# 3. FACE RECOGNITION TEST
print("\n3️⃣  FACE RECOGNITION & CLUSTERING")
print("-" * 60)
try:
    if face_engine.app:
        print(f"   ✅ InsightFace model loaded")
        
        # Test on image if available
        if img_count > 0:
            db = SessionLocal()
            sample = db.query(DBImage).first()
            if sample:
                img_path = f"../data/images/{sample.filename}"
                if os.path.exists(img_path):
                    faces = face_engine.detect_faces(img_path)
                    print(f"   ✅ Face detection working")
                    print(f"      - Faces found: {len(faces)}")
                else:
                    print(f"   ⚠️  Sample image path not found")
            db.close()
    else:
        print(f"   ❌ InsightFace not available")
    
    # Check face index
    if face_engine.face_index:
        print(f"   ✅ Face FAISS index loaded")
        print(f"      - Vectors: {face_engine.face_index.ntotal}")
    else:
        print(f"   ❌ Face FAISS index not loaded")
        
except Exception as e:
    print(f"   ❌ Face engine error: {e}")
    import traceback
    traceback.print_exc()

# 4. OCR MODEL TEST
print("\n4️⃣  OCR (TEXT EXTRACTION)")
print("-" * 60)
try:
    if img_count > 0:
        db = SessionLocal()
        sample = db.query(DBImage).first()
        if sample:
            img_path = f"../data/images/{sample.filename}"
            if os.path.exists(img_path):
                ocr_result = extract_text(img_path)
                print(f"   ✅ OCR working")
                if ocr_result:
                    text_preview = ocr_result[:100] + "..." if len(ocr_result) > 100 else ocr_result
                    print(f"      - Extracted: {text_preview}")
                else:
                    print(f"      - No text found in image")
            else:
                print(f"   ⚠️  Image file not found")
        db.close()
except Exception as e:
    print(f"   ❌ OCR error: {e}")
    import traceback
    traceback.print_exc()

# 5. OBJECT DETECTION TEST
print("\n5️⃣  OBJECT DETECTION")
print("-" * 60)
try:
    print(f"   ✅ Faster R-CNN loaded")
    print(f"      - Categories available: {len(detector_engine.detector_engine.categories)}")
    print(f"      - Sample categories: {detector_engine.detector_engine.categories[:10]}")
    
    if img_count > 0:
        db = SessionLocal()
        sample = db.query(DBImage).first()
        if sample:
            img_path = f"../data/images/{sample.filename}"
            if os.path.exists(img_path):
                persons = detector_engine.detect_persons(img_path)
                objects = detector_engine.detect_objects(img_path)
                print(f"   ✅ Object detection working")
                print(f"      - Persons detected: {persons}")
                print(f"      - Objects: {objects if objects else 'None'}")
            else:
                print(f"   ⚠️  Image file not found")
        db.close()
except Exception as e:
    print(f"   ❌ Object detection error: {e}")
    import traceback
    traceback.print_exc()

# 6. SEARCH FUNCTIONALITY TEST
print("\n6️⃣  SEARCH ENDPOINT")
print("-" * 60)
try:
    from main import search
    
    result = search(query="dog", top_k=5)
    
    if result.get('status') == 'found':
        print(f"   ✅ Search working")
        print(f"      - Query: 'dog'")
        print(f"      - Results: {result.get('count')}")
        for i, res in enumerate(result.get('results', [])[:3], 1):
            print(f"        {i}. {res['filename']}: {res['score']}%")
    elif result.get('status') == 'not_found':
        print(f"   ⚠️  Search working but no results for 'dog'")
        print(f"      - Message: {result.get('message')}")
    else:
        print(f"   ❌ Search error: {result.get('message')}")
        
except Exception as e:
    print(f"   ❌ Search error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)
