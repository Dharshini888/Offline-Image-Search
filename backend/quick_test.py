#!/usr/bin/env python3
"""
Quick Test Suite - Copy to backend folder and run:
    python quick_test.py
"""

import os
import sys

def test_1_clip_model():
    """Test if CLIP model embeddings work"""
    print("\n" + "="*70)
    print("TEST 1: Is CLIP Model Working?")
    print("="*70)
    
    try:
        import torch
        import clip
        from PIL import Image
        
        print("✓ Loading CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        
        # Find an image
        IMAGE_DIR = "../data/images"
        if not os.path.exists(IMAGE_DIR):
            print("❌ Image folder not found")
            return False
        
        images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            print("❌ No images found")
            return False
        
        # Test embedding
        img_path = os.path.join(IMAGE_DIR, images[0])
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            img_emb = model.encode_image(img_tensor)
        
        print(f"✓ Image embedding: {img_emb.shape}")
        
        # Test text embedding
        text = clip.tokenize(["a photo of a dog"])
        with torch.no_grad():
            text_emb = model.encode_text(text)
        
        print(f"✓ Text embedding: {text_emb.shape}")
        print("\n✅ CLIP Model is WORKING!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ CLIP Model FAILED: {e}\n")
        return False

def test_2_search():
    """Test if search returns results"""
    print("="*70)
    print("TEST 2: Does Search Work?")
    print("="*70)
    
    try:
        import sqlite3
        
        db_path = "../data/db.sqlite"
        if not os.path.exists(db_path):
            print("❌ Database not found - run build_index.py first\n")
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM images")
        img_count = cursor.fetchone()[0]
        
        print(f"✓ Images in database: {img_count}")
        
        if img_count == 0:
            print("❌ No images indexed\n")
            return False
        
        # Check FAISS index
        if not os.path.exists("../data/index.faiss"):
            print("❌ FAISS index not found\n")
            return False
        
        import faiss
        index = faiss.read_index("../data/index.faiss")
        print(f"✓ FAISS index has {index.ntotal} vectors")
        
        print("\n✅ Search Index is WORKING!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Search FAILED: {e}\n")
        return False

def test_3_clustering():
    """Test if face clustering worked"""
    print("="*70)
    print("TEST 3: Did Face Clustering Work?")
    print("="*70)
    
    try:
        import sqlite3
        
        db_path = "../data/db.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM people")
        people_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM faces")
        face_count = cursor.fetchone()[0]
        
        print(f"✓ Faces detected: {face_count}")
        print(f"✓ People clusters: {people_count}")
        
        if people_count == 0 and face_count > 0:
            print("\n⚠️  Clustering not run yet!")
            print("   Solution: curl -X POST http://localhost:8000/recluster")
            return False
        
        print("\n✅ Clustering is WORKING!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Clustering check FAILED: {e}\n")
        return False

def test_4_albums():
    """Test if albums were created"""
    print("="*70)
    print("TEST 4: Were Albums/Events Created?")
    print("="*70)
    
    try:
        import sqlite3
        
        db_path = "../data/db.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM albums")
        album_count = cursor.fetchone()[0]
        
        print(f"✓ Albums created: {album_count}")
        
        if album_count == 0:
            print("\n⚠️  No albums created!")
            print("   Solution: curl -X POST http://localhost:8000/recluster")
            return False
        
        print("\n✅ Albums are WORKING!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Albums check FAILED: {e}\n")
        return False

def main():
    print("\n" + "="*70)
    print("🧪 QUICK TEST SUITE")
    print("="*70)
    print("This will test all major components\n")
    
    results = {
        "CLIP Model": test_1_clip_model(),
        "Search Index": test_2_search(),
        "Clustering": test_3_clustering(),
        "Albums": test_4_albums(),
    }
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    passed = sum(1 for p in results.values() if p)
    total = len(results)
    
    print(f"\nResult: {passed}/{total} tests passed\n")
    
    if passed < total:
        print("⚠️  Some tests failed. Follow these steps:\n")
        
        if not results.get("CLIP Model"):
            print("1. Check CLIP is installed:")
            print("   pip install torch torchvision clip")
        
        if not results.get("Search Index"):
            print("2. Rebuild search index:")
            print("   del ..\\data\\db.sqlite")
            print("   del ..\\data\\index.faiss")
            print("   python build_index.py")
        
        if not results.get("Clustering"):
            print("3. Run clustering:")
            print("   curl -X POST http://localhost:8000/recluster")
        
        if not results.get("Albums"):
            print("4. Clustering needed for albums:")
            print("   curl -X POST http://localhost:8000/recluster")

if __name__ == "__main__":
    main()