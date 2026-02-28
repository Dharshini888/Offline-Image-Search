"""
Comprehensive diagnostic script to test all models and identify issues.
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_clip_model():
    """Test CLIP model loading and embedding generation"""
    logger.info("=" * 60)
    logger.info("Testing CLIP Model")
    logger.info("=" * 60)
    
    try:
        from search_engine import search_engine
        
        # Test text embedding
        text_emb = search_engine.get_text_embedding("dog", use_prompt_ensemble=True)
        if text_emb is not None:
            logger.info(f"✅ CLIP text embedding: shape={text_emb.shape}, norm={np.linalg.norm(text_emb):.4f}")
        else:
            logger.error("❌ CLIP text embedding failed")
            return False
        
        # Test image embedding
        image_dir = Path("../data/images")
        if image_dir.exists():
            images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
            if images:
                test_img = str(images[0])
                img_emb = search_engine.get_image_embedding(test_img)
                if img_emb is not None:
                    logger.info(f"✅ CLIP image embedding: shape={img_emb.shape}, norm={np.linalg.norm(img_emb):.4f}")
                    
                    # Test similarity
                    similarity = np.dot(text_emb, img_emb)
                    logger.info(f"✅ Text-Image similarity: {similarity:.4f}")
                else:
                    logger.error("❌ CLIP image embedding failed")
                    return False
            else:
                logger.warning("⚠️  No images found for testing")
        
        return True
    except Exception as e:
        logger.error(f"❌ CLIP model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_recognition():
    """Test face recognition model"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Face Recognition (InsightFace)")
    logger.info("=" * 60)
    
    try:
        from face_engine import face_engine, INSIGHTFACE_AVAILABLE
        
        if not INSIGHTFACE_AVAILABLE:
            logger.error("❌ InsightFace not available")
            return False
        
        if face_engine.app is None:
            logger.error("❌ Face engine not initialized")
            return False
        
        logger.info("✅ InsightFace model loaded")
        
        # Test face detection
        image_dir = Path("../data/images")
        if image_dir.exists():
            images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
            if images:
                test_img = str(images[0])
                faces = face_engine.detect_faces(test_img)
                logger.info(f"✅ Face detection: found {len(faces)} faces in {images[0].name}")
                
                if faces:
                    logger.info(f"   First face embedding shape: {faces[0]['embedding'].shape}")
                    logger.info(f"   First face bbox: {faces[0]['bbox']}")
            else:
                logger.warning("⚠️  No images found for testing")
        
        # Test clustering
        if face_engine.face_index:
            logger.info(f"✅ Face FAISS index: {face_engine.face_index.ntotal} vectors")
        else:
            logger.warning("⚠️  Face FAISS index not loaded")
        
        return True
    except Exception as e:
        logger.error(f"❌ Face recognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_object_detection():
    """Test Faster R-CNN object detection"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Object Detection (Faster R-CNN)")
    logger.info("=" * 60)
    
    try:
        from detector_engine import detector_engine
        
        logger.info(f"✅ Faster R-CNN loaded on {detector_engine.model.training}")
        logger.info(f"   Categories available: {len(detector_engine.categories)}")
        
        # Test object detection
        image_dir = Path("../data/images")
        if image_dir.exists():
            images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
            if images:
                test_img = str(images[0])
                objects = detector_engine.detect_objects(test_img, threshold=0.5)
                logger.info(f"✅ Object detection: found {len(objects)} objects in {images[0].name}")
                if objects:
                    logger.info(f"   Objects: {', '.join(objects)}")
                
                person_count = detector_engine.detect_persons(test_img)
                logger.info(f"✅ Person detection: found {person_count} persons")
            else:
                logger.warning("⚠️  No images found for testing")
        
        return True
    except Exception as e:
        logger.error(f"❌ Object detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ocr():
    """Test OCR (Tesseract)"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing OCR (Tesseract)")
    logger.info("=" * 60)
    
    try:
        from ocr_engine import extract_text
        
        # Test OCR
        image_dir = Path("../data/images")
        if image_dir.exists():
            images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
            if images:
                test_img = str(images[0])
                text = extract_text(test_img)
                if text:
                    logger.info(f"✅ OCR extracted text: '{text[:100]}...'")
                else:
                    logger.info("✅ OCR working (no text found in image)")
            else:
                logger.warning("⚠️  No images found for testing")
        
        return True
    except Exception as e:
        logger.error(f"❌ OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database():
    """Test database connectivity and schema"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Database")
    logger.info("=" * 60)
    
    try:
        from database import SessionLocal, Image as DBImage, Face as DBFace, Person, init_db
        
        init_db()
        logger.info("✅ Database initialized")
        
        db = SessionLocal()
        
        # Count records
        image_count = db.query(DBImage).count()
        face_count = db.query(DBFace).count()
        person_count = db.query(Person).count()
        
        logger.info(f"✅ Database records:")
        logger.info(f"   Images: {image_count}")
        logger.info(f"   Faces: {face_count}")
        logger.info(f"   People: {person_count}")
        
        # Check schema
        if image_count > 0:
            sample = db.query(DBImage).first()
            logger.info(f"✅ Sample image:")
            logger.info(f"   Filename: {sample.filename}")
            logger.info(f"   Scene label: {sample.scene_label}")
            logger.info(f"   Person count: {sample.person_count}")
            logger.info(f"   OCR text: {sample.ocr_text[:50] if sample.ocr_text else 'None'}...")
            logger.info(f"   Avg color: R={sample.avg_r:.1f}, G={sample.avg_g:.1f}, B={sample.avg_b:.1f}")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_indexes():
    """Test FAISS indexes"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing FAISS Indexes")
    logger.info("=" * 60)
    
    try:
        import faiss
        
        # Test image index
        image_index_path = "../data/index.faiss"
        if os.path.exists(image_index_path):
            index = faiss.read_index(image_index_path)
            logger.info(f"✅ Image FAISS index: {index.ntotal} vectors")
        else:
            logger.warning("⚠️  Image FAISS index not found")
        
        # Test face index
        face_index_path = "../data/face_index.faiss"
        if os.path.exists(face_index_path):
            face_index = faiss.read_index(face_index_path)
            logger.info(f"✅ Face FAISS index: {face_index.ntotal} vectors")
        else:
            logger.warning("⚠️  Face FAISS index not found")
        
        return True
    except Exception as e:
        logger.error(f"❌ FAISS index test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_functionality():
    """Test search with different queries"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Search Functionality")
    logger.info("=" * 60)
    
    try:
        from search_engine import search_engine, resolve_query
        from database import SessionLocal, Image as DBImage
        import faiss
        
        # Load index
        image_index_path = "../data/index.faiss"
        if not os.path.exists(image_index_path):
            logger.warning("⚠️  No FAISS index found, skipping search test")
            return True
        
        search_engine.index = faiss.read_index(image_index_path)
        
        test_queries = ["dog", "person", "car", "food", "sunset"]
        
        db = SessionLocal()
        
        for query in test_queries:
            processed = resolve_query(query)
            logger.info(f"\n🔍 Query: '{query}' → '{processed}'")
            
            query_emb = search_engine.get_text_embedding(processed, use_prompt_ensemble=True)
            if query_emb is None:
                continue
            
            query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_emb_reshaped)
            
            distances, indices = search_engine.index.search(query_emb_reshaped, 5)
            
            logger.info(f"   Top 5 results:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                
                img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
                if img:
                    clip_score = max(0.0, min(1.0, (float(dist) + 1.0) / 2.0))
                    logger.info(f"   {i+1}. {img.filename} - CLIP score: {clip_score:.3f}")
                    logger.info(f"      Scene: {img.scene_label}, Persons: {img.person_count}")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE DIAGNOSTIC TEST")
    logger.info("=" * 60 + "\n")
    
    results = {
        "CLIP Model": test_clip_model(),
        "Face Recognition": test_face_recognition(),
        "Object Detection": test_object_detection(),
        "OCR": test_ocr(),
        "Database": test_database(),
        "FAISS Indexes": test_faiss_indexes(),
        "Search Functionality": test_search_functionality(),
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.info("\n⚠️  Some tests failed. Please check the logs above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
