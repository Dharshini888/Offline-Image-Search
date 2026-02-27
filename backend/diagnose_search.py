#!/usr/bin/env python3
"""
Diagnose Search Quality Issues
Run: python diagnose_search.py
"""

import os
import sys

def main():
    print("\n" + "="*80)
    print("🔍 SEARCH QUALITY DIAGNOSTIC")
    print("="*80)
    
    # Check prerequisites
    print("\n1️⃣  Checking prerequisites...")
    
    if not os.path.exists("../data/db.sqlite"):
        print("   ❌ Database not found - run build_index.py first")
        return
    
    if not os.path.exists("../data/index.faiss"):
        print("   ❌ FAISS index not found - run build_index.py first")
        return
    
    print("   ✅ Database and index found")
    
    # Load components
    print("\n2️⃣  Loading components...")
    
    try:
        from search_engine import search_engine
        from database import SessionLocal, Image as DBImage, init_db
        import faiss
        import numpy as np
        
        print("   ✅ Modules loaded")
    except Exception as e:
        print(f"   ❌ Failed to load modules: {e}")
        return
    
    # Load index
    try:
        search_engine.index = faiss.read_index("../data/index.faiss")
        print(f"   ✅ FAISS index loaded ({search_engine.index.ntotal} vectors)")
    except Exception as e:
        print(f"   ❌ Failed to load FAISS: {e}")
        return
    
    # Test search
    print("\n3️⃣  Testing search quality...")
    
    init_db()
    db = SessionLocal()
    
    test_queries = ["dog", "cat", "person", "outdoor"]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        try:
            # Get embedding
            emb = search_engine.get_text_embedding(query, use_prompt_ensemble=True)
            
            if emb is None:
                print(f"      ❌ Could not get embedding")
                continue
            
            # Search
            q = emb.reshape(1, -1).astype('float32')
            faiss.normalize_L2(q)
            dists, idxs = search_engine.index.search(q, 5)
            
            valid = 0
            for sim, idx in zip(dists[0], idxs[0]):
                if idx == -1:
                    continue
                
                img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
                if not img:
                    continue
                
                clip_score = float(sim)
                
                # Calculate final score
                final = clip_score * 0.75  # Simplified
                
                status = "✅" if clip_score > 0.35 else "❌"
                valid += 1 if clip_score > 0.35 else 0
                
                print(f"      {status} {img.filename[:30]:<30} score={clip_score:.3f}")
            
            if valid == 0:
                print(f"      ⚠️  No results passed threshold (0.35)")
                print(f"         Solution: Lower CLIP_SCORE_MIN in debug_search.py")
        
        except Exception as e:
            print(f"      ❌ Search failed: {e}")
    
    # Check thresholds
    print(f"\n4️⃣  Current Thresholds")
    print("="*80)
    
    try:
        with open("debug_search.py", "r") as f:
            content = f.read()
            
            if "CLIP_SCORE_MIN = 0.20" in content:
                print("   ⚠️  CLIP_SCORE_MIN = 0.20 (might be too low)")
            elif "CLIP_SCORE_MIN = 0.30" in content:
                print("   ✅ CLIP_SCORE_MIN = 0.30 (good)")
            elif "CLIP_SCORE_MIN = 0.35" in content:
                print("   ✅ CLIP_SCORE_MIN = 0.35 (very strict)")
            
            if "FINAL_SCORE_MIN = 0.25" in content:
                print("   ⚠️  FINAL_SCORE_MIN = 0.25 (might be too low)")
            elif "FINAL_SCORE_MIN = 0.35" in content:
                print("   ✅ FINAL_SCORE_MIN = 0.35 (good)")
            elif "FINAL_SCORE_MIN = 0.40" in content:
                print("   ✅ FINAL_SCORE_MIN = 0.40 (very strict)")
    except:
        print("   ℹ️  Could not check thresholds")
    
    # Check weights
    print(f"\n5️⃣  Ranking Weights")
    print("="*80)
    
    try:
        with open("search_engine.py", "r") as f:
            content = f.read()
            
            if "(0.60 * clip_score)" in content:
                print("   ⚠️  CLIP weight = 0.60 (allows OCR/tags to override)")
            elif "(0.75 * clip_score)" in content:
                print("   ✅ CLIP weight = 0.75 (CLIP dominates)")
            elif "(0.80 * clip_score)" in content:
                print("   ✅ CLIP weight = 0.80 (very strict)")
    except:
        print("   ℹ️  Could not check weights")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    print("""
If search returns unrelated images:

1. HIGHEST PRIORITY - Raise thresholds:
   Edit debug_search.py lines 8-9:
   - CLIP_SCORE_MIN = 0.35 (was 0.20)
   - FINAL_SCORE_MIN = 0.40 (was 0.25)

2. If still getting unrelated images:
   Edit search_engine.py hybrid_rank():
   - Increase CLIP weight from 0.60 to 0.80
   - Decrease OCR weight from 0.20 to 0.10

3. If dog images don't return:
   - Your images might be poor quality
   - Try: CLIP_SCORE_MIN = 0.20 (be more permissive)
   - Or add more/better quality images

4. To test the fix:
   python diagnose_search.py
   python debug_search_detailed.py dog
""")
    
    db.close()

if __name__ == "__main__":
    main()