#!/usr/bin/env python3
import os
import sys
os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, '.')

from database import SessionLocal, init_db, Image as DBImage, Face as DBFace, Person

init_db()
db = SessionLocal()

try:
    img_count = db.query(DBImage).count()
    face_count = db.query(DBFace).count()
    person_count = db.query(Person).count()
    
    print(f'✅ Database ready:')
    print(f'  Images: {img_count}')
    print(f'  Faces: {face_count}')
    print(f'  People (clusters): {person_count}')
    
    # Check if index file exists
    if os.path.exists('../data/index.faiss'):
        print(f'  ✅ Image FAISS index exists')
    else:
        print(f'  ❌ Image FAISS index missing')
    
    if os.path.exists('../data/face_index.faiss'):
        print(f'  ✅ Face FAISS index exists')
    else:
        print(f'  ❌ Face FAISS index missing')
    
    # Sample images
    print(f'\nSample images:')
    imgs = db.query(DBImage).limit(3).all()
    for img in imgs:
        faces = db.query(DBFace).filter(DBFace.image_id == img.id).count()
        print(f'  - {img.filename}: {faces} faces, tags={img.scene_label}')
        
finally:
    db.close()
