"""
Quick test script to verify search functionality and model performance
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_search_accuracy():
    """Test search with specific queries and verify results"""
    print("=" * 60)
    print("TESTING SEARCH ACCURACY")
    print("=" * 60)
    
    from database import SessionLocal, Image as DBImage
    from search_engine import search_engine, resolve_query
    import faiss
    import numpy as np
    
    # Load index
    index_path = "data/index.faiss"
    if not os.path.exists(index_path):
        print("❌ No FAISS index found. Run build_index.py first!")
        return False
    
    search_engine.index = faiss.read_index(index_path)
    print(f"✅ Loaded FAISS index with {search_engine.index.ntotal} vectors\n")
    
    db = SessionLocal()
    
    # Test queries
    test_cases = [
        {
            "query": "dog",
            "expected_objects": ["dog"],
            "should_not_contain": ["person only"]
        },
        {
            "query": "person",
            "expected_objects": ["person"],
            "should_not_contain": []
        },
        {
            "query": "car",
            "expected_objects": ["car"],
            "should_not_contain": []
        }
    ]
    
    for test in test_cases:
        query = test["query"]
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 60)
        
        # Get embedding
        processed = resolve_query(query)
        query_emb = search_engine.get_text_embedding(processed, use_prompt_ensemble=True)
        
        if query_emb is None:
            print(f"❌ Failed to get embedding for '{query}'")
            continue
        
        # Search
        query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb_reshaped)
        distances, indices = search_engine.index.search(query_emb_reshaped, 10)
        
        # Analyze results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
            if not img:
                continue
            
            # Calculate scores
            raw_sim = float(dist)
            clip_score = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))
            
            # Get object tags
            tags = []
            if img.scene_label:
                tags = [t.strip().lower() for t in img.scene_label.split(",")]
            
            results.append({
                "filename": img.filename,
                "clip_score": clip_score,
                "tags": tags,
                "person_count": img.person_count or 0
            })
        
        # Display results
        print(f"Found {len(results)} results:")
        for i, r in enumerate(results[:5], 1):
            print(f"  {i}. {r['filename']}")
            print(f"     CLIP: {r['clip_score']:.3f} | Tags: {', '.join(r['tags']) if r['tags'] else 'none'} | Persons: {r['person_count']}")
        
        # Verify accuracy
        if results:
            # Check if expected objects are in top results
            top_tags = set()
            for r in results[:5]:
                top_tags.update(r['tags'])
            
            expected = test["expected_objects"]
            found_expected = any(exp in top_tags for exp in expected)
            
            if found_expected:
                print(f"✅ PASS - Found expected objects: {expected}")
            else:
                print(f"⚠️  WARNING - Expected objects not in top results: {expected}")
                print(f"   Found tags: {top_tags}")
        else:
            print(f"❌ FAIL - No results found")
    
    db.close()
    return True

def test_face_clustering():
    """Test face clustering quality"""
    print("\n" + "=" * 60)
    print("TESTING FACE CLUSTERING")
    print("=" * 60)
    
    from database import SessionLocal, Person, Face as DBFace
    
    db = SessionLocal()
    
    people = db.query(Person).all()
    print(f"\n✅ Found {len(people)} people in database\n")
    
    for person in people[:10]:  # Show first 10
        faces = db.query(DBFace).filter(DBFace.person_id == person.id).all()
        print(f"  {person.name}: {len(faces)} faces")
    
    if len(people) == 0:
        print("⚠️  No people found. Face clustering may not have run.")
        print("   Run: python backend/build_index.py")
    
    db.close()
    return True

def test_object_detection():
    """Test object detection on sample images"""
    print("\n" + "=" * 60)
    print("TESTING OBJECT DETECTION")
    print("=" * 60)
    
    from database import SessionLocal, Image as DBImage
    
    db = SessionLocal()
    
    images = db.query(DBImage).limit(10).all()
    
    print(f"\n✅ Checking object tags for {len(images)} sample images:\n")
    
    for img in images:
        tags = img.scene_label.split(",") if img.scene_label else []
        print(f"  {img.filename}")
        print(f"    Tags: {', '.join(tags) if tags else 'none'}")
        print(f"    Persons: {img.person_count or 0}")
    
    # Count images with tags
    images_with_tags = db.query(DBImage).filter(DBImage.scene_label != None).count()
    total_images = db.query(DBImage).count()
    
    print(f"\n📊 Statistics:")
    print(f"   Images with object tags: {images_with_tags}/{total_images}")
    
    if images_with_tags == 0:
        print("⚠️  No object tags found. Object detection may not have run.")
        print("   Run: python backend/build_index.py")
    
    db.close()
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SMARTGALLERY - QUICK TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Search Accuracy
        test_search_accuracy()
        
        # Test 2: Face Clustering
        test_face_clustering()
        
        # Test 3: Object Detection
        test_object_detection()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. If object tags are missing, run: python backend/build_index.py")
        print("2. If search results are poor, adjust thresholds in main.py")
        print("3. If faces aren't clustered, check InsightFace installation")
        print("\nFor detailed diagnostics, run: python backend/comprehensive_diagnostic.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
