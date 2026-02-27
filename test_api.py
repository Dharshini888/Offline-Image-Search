#!/usr/bin/env python3
"""Quick API test to debug issues"""
import sys
import os

# Change to backend directory
os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, '.')

from database import SessionLocal, Image as DBImage
from search_engine import search_engine

db = SessionLocal()
try:
    # Test 1: Check if database has images
    count = db.query(DBImage).count()
    print(f"✅ Database connected. Found {count} images.")
    
    # Test 2: Try to fetch images like timeline does
    images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
    print(f"✅ Retrieved {len(images)} images from database")
    
    if images:
        img = images[0]
        print(f"  Sample: {img.filename} - {img.timestamp} - tags: {img.scene_label}")
    
    # Test 3: Check if search index is loaded
    print(f"✅ FAISS index: {search_engine.index.ntotal if search_engine.index else 0} vectors")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
