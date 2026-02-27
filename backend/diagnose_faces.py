#!/usr/bin/env python3
"""
Diagnose Face Clustering Issues
Run: python diagnose_faces.py
"""

import sqlite3
import os

def main():
    print("\n" + "="*80)
    print("🔍 FACE CLUSTERING DIAGNOSTIC")
    print("="*80)
    
    db_path = "../data/db.sqlite"
    
    if not os.path.exists(db_path):
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get stats
    cursor.execute("SELECT COUNT(*) FROM faces")
    face_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM people")
    people_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM images")
    image_count = cursor.fetchone()[0]
    
    print(f"\nDatabase Stats:")
    print(f"  Total images: {image_count}")
    print(f"  Total faces: {face_count}")
    print(f"  Total people: {people_count}")
    
    # Find images with multiple faces
    print(f"\n{'='*80}")
    print("🔎 Checking Images with Multiple Faces")
    print("="*80)
    
    cursor.execute("""
        SELECT img.id, img.filename, COUNT(f.id) as face_count
        FROM images img
        LEFT JOIN faces f ON img.id = f.image_id
        GROUP BY img.id
        HAVING face_count > 1
        ORDER BY face_count DESC
    """)
    
    multi_face_images = cursor.fetchall()
    
    if not multi_face_images:
        print("\n✅ No images with multiple faces found (or all faces filtered out)")
    else:
        print(f"\nFound {len(multi_face_images)} images with multiple faces:\n")
        
        problem_count = 0
        
        for img_id, filename, face_count in multi_face_images[:10]:
            # Get people for these faces
            cursor.execute("""
                SELECT f.id, f.person_id, p.name
                FROM faces f
                LEFT JOIN people p ON f.person_id = p.id
                WHERE f.image_id = ?
                ORDER BY f.id
            """, (img_id,))
            
            faces = cursor.fetchall()
            person_ids = [f[1] for f in faces]
            unique_people = len(set(p for p in person_ids if p is not None))
            
            print(f"📷 {filename}")
            print(f"   Faces: {face_count}, People: {unique_people}")
            
            if unique_people > 1:
                print(f"   ❌ PROBLEM: Same image assigned to {unique_people} different people!")
                problem_count += 1
                
                # Show each face
                for face_id, person_id, person_name in faces:
                    print(f"      Face {face_id} → {person_name or 'Unassigned'}")
            else:
                print(f"   ✅ Correct: All faces assigned to same person")
            
            print()
        
        if problem_count == 0:
            print(f"✅ All {len(multi_face_images)} images correctly clustered!")
        else:
            print(f"\n❌ {problem_count} images have clustering problems")
            print("\nHow to fix:")
            print("  1. Edit face_engine.py, find cluster_faces() function")
            print("  2. Change: model = DBSCAN(eps=0.5, min_samples=3, ...)")
            print("  3. To:     model = DBSCAN(eps=0.35, min_samples=2, ...)")
            print("  4. Delete: del ..\\data\\db.sqlite .fa face_index.faiss")
            print("  5. Rebuild: python build_index.py")
    
    # Check unassigned faces
    print(f"\n{'='*80}")
    print("🔗 Checking Unassigned Faces")
    print("="*80)
    
    cursor.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NULL")
    unassigned = cursor.fetchone()[0]
    
    total_faces = face_count
    assigned = total_faces - unassigned
    pct = (assigned / total_faces * 100) if total_faces > 0 else 0
    
    print(f"\nFaces assigned to people: {assigned}/{total_faces} ({pct:.1f}%)")
    
    if pct < 50:
        print("⚠️  WARNING: Less than 50% of faces assigned!")
        print("   Solution: Run: curl -X POST http://localhost:8000/recluster")
    elif pct == 100:
        print("✅ All faces successfully clustered")
    else:
        print(f"✅ {pct:.1f}% faces clustered (acceptable)")
    
    # Check for duplicate people with same faces
    print(f"\n{'='*80}")
    print("🧑‍🤝 Checking for Duplicate People")
    print("="*80)
    
    cursor.execute("""
        SELECT p1.id, p1.name, COUNT(*) as face_count
        FROM people p1
        LEFT JOIN faces f ON p1.id = f.person_id
        GROUP BY p1.id
        ORDER BY face_count DESC
        LIMIT 5
    """)
    
    people = cursor.fetchall()
    
    print(f"\nTop people by face count:\n")
    for person_id, name, count in people:
        print(f"  {name}: {count} faces")
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 SUMMARY")
    print("="*80)
    
    print(f"\n✅ Total images: {image_count}")
    print(f"✅ Total faces: {face_count}")
    print(f"✅ Total people: {people_count}")
    print(f"✅ Faces assigned: {assigned}/{total_faces} ({pct:.1f}%)")
    
    if problem_count > 0:
        print(f"\n❌ Found {problem_count} images with incorrect clustering")
        print("   This causes multiple people to be assigned to the same image")
    elif people_count == 0:
        print(f"\n⚠️  No clustering has been run yet")
        print("   Run: curl -X POST http://localhost:8000/recluster")
    else:
        print(f"\n✅ Clustering appears to be working correctly!")
    
    conn.close()

if __name__ == "__main__":
    main()