
# # # # # # # # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# # # # # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # # # # from fastapi.staticfiles import StaticFiles
# # # # # # # # import os
# # # # # # # # import uuid
# # # # # # # # import shutil
# # # # # # # # import numpy as np
# # # # # # # # import faiss
# # # # # # # # from datetime import datetime
# # # # # # # # import logging
# # # # # # # # from contextlib import asynccontextmanager

# # # # # # # # # Setup Logging
# # # # # # # # logging.basicConfig(level=logging.INFO)
# # # # # # # # logger = logging.getLogger("main")

# # # # # # # # from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
# # # # # # # # from search_engine import search_engine, resolve_query
# # # # # # # # from voice_engine import voice_engine
# # # # # # # # from face_engine import face_engine
# # # # # # # # from ocr_engine import extract_text
# # # # # # # # from detector_engine import detector_engine
# # # # # # # # from duplicate_engine import duplicate_engine

# # # # # # # # # Paths
# # # # # # # # IMAGE_DIR = "../data/images"
# # # # # # # # FAISS_INDEX_PATH = "../data/index.faiss"

# # # # # # # # # Configurable thresholds and options
# # # # # # # # FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
# # # # # # # # FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
# # # # # # # # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # # # # # # # RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # # # # # # # RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # # # # # # # # Search thresholds - SMART COMBINED SCORING
# # # # # # # # CLIP_SCORE_MIN = 0.10      # Very low - let combined score decide
# # # # # # # # FINAL_SCORE_MIN = 0.15     # Combined score threshold
# # # # # # # # SEARCH_SCORE_THRESHOLD = float(os.environ.get("SEARCH_SCORE_THRESHOLD", 0.20))
# # # # # # # # RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # # # # # # # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # # # # # # # recluster_last_triggered = None  # Track last recluster time for debouncing

# # # # # # # # def should_trigger_recluster(background_tasks):
# # # # # # # #     """Check if batch size reached or timer expired; schedule recluster if needed."""
# # # # # # # #     global recluster_last_triggered
    
# # # # # # # #     if not RECLUSTER_ON_UPLOAD or not background_tasks:
# # # # # # # #         return
    
# # # # # # # #     try:
# # # # # # # #         # Increment counter
# # # # # # # #         counter = 0
# # # # # # # #         if os.path.exists(RECLUSTER_COUNTER_PATH):
# # # # # # # #             try:
# # # # # # # #                 with open(RECLUSTER_COUNTER_PATH, 'r') as f:
# # # # # # # #                     counter = int(f.read().strip())
# # # # # # # #             except: pass
        
# # # # # # # #         counter += 1
# # # # # # # #         with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # # # # # #             f.write(str(counter))
        
# # # # # # # #         # Check if batch size reached
# # # # # # # #         should_trigger = counter >= RECLUSTER_BATCH_SIZE
        
# # # # # # # #         # Also check if timer expired since last recluster
# # # # # # # #         now = datetime.now()
# # # # # # # #         if recluster_last_triggered:
# # # # # # # #             elapsed = (now - recluster_last_triggered).total_seconds()
# # # # # # # #             if elapsed >= RECLUSTER_TIMER_SECONDS:
# # # # # # # #                 should_trigger = True
# # # # # # # #         elif counter > 0:
# # # # # # # #             should_trigger = counter >= RECLUSTER_BATCH_SIZE
        
# # # # # # # #         if should_trigger:
# # # # # # # #             logger.info(f"📊 Recluster triggered: counter={counter}, batch_size={RECLUSTER_BATCH_SIZE}")
# # # # # # # #             background_tasks.add_task(recluster)
# # # # # # # #             recluster_last_triggered = now
# # # # # # # #             # Reset counter
# # # # # # # #             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # # # # # #                 f.write('0')
# # # # # # # #     except Exception as e:
# # # # # # # #         logger.warning(f"Batched recluster check failed: {e}")

# # # # # # # # @asynccontextmanager
# # # # # # # # async def lifespan(app: FastAPI):
# # # # # # # #     # Initialize DB
# # # # # # # #     init_db()
# # # # # # # #     logger.info("🚀 Starting Offline Smart Gallery Backend...")

# # # # # # # #     # Load image FAISS index
# # # # # # # #     if os.path.exists(FAISS_INDEX_PATH):
# # # # # # # #         try:
# # # # # # # #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# # # # # # # #             logger.info(f"✅ Image FAISS index loaded ({search_engine.index.ntotal} vectors).")
# # # # # # # #         except Exception as e:
# # # # # # # #             logger.error(f"❌ Error loading image FAISS index: {e}")
# # # # # # # #             search_engine.index = None
# # # # # # # #     else:
# # # # # # # #         logger.warning("⚠️  Image FAISS index not found. Searching will be limited.")

# # # # # # # #     # Load face FAISS index (handled by face_engine internally)
# # # # # # # #     logger.info(f"✅ Face FAISS index ready ({face_engine.face_index.ntotal if face_engine.face_index else 0} vectors).")
# # # # # # # #     yield

# # # # # # # # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)

# # # # # # # # app.add_middleware(
# # # # # # # #     CORSMiddleware,
# # # # # # # #     allow_origins=["*"],
# # # # # # # #     allow_methods=["*"],
# # # # # # # #     allow_headers=["*"],
# # # # # # # # )

# # # # # # # # if not os.path.exists(IMAGE_DIR):
# # # # # # # #     os.makedirs(IMAGE_DIR)

# # # # # # # # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # # # # # # # @app.get("/health")
# # # # # # # # def health():
# # # # # # # #     return {
# # # # # # # #         "status": "ready",
# # # # # # # #         "models": ["CLIP", "OCR", "FaceRecognition", "Clustering"],
# # # # # # # #         "image_index": search_engine.index.ntotal if search_engine.index else 0,
# # # # # # # #         "face_index": face_engine.face_index.ntotal if face_engine.face_index else 0
# # # # # # # #     }

# # # # # # # # @app.get("/test-db")
# # # # # # # # def test_db():
# # # # # # # #     """Diagnostic endpoint to test database"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         count = db.query(DBImage).count()
# # # # # # # #         images = db.query(DBImage).limit(1).all()
# # # # # # # #         return {
# # # # # # # #             "status": "ok",
# # # # # # # #             "total_images": count,
# # # # # # # #             "sample": {
# # # # # # # #                 "filename": images[0].filename,
# # # # # # # #                 "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None
# # # # # # # #             } if images else None
# # # # # # # #         }
# # # # # # # #     except Exception as e:
# # # # # # # #         logger.error(f"❌ Test DB error: {e}", exc_info=True)
# # # # # # # #         return {"status": "error", "message": str(e)}
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.post("/search")
# # # # # # # # def search(query: str = Form(...), top_k: int = Form(20)):
# # # # # # # #     """
# # # # # # # #     Search endpoint with SMART COMBINED SCORING:
# # # # # # # #     - Uses OCR and object tags to boost relevant results
# # # # # # # #     - Rejects random unrelated images
# # # # # # # #     - Only images matching keywords AND having decent CLIP score are shown
# # # # # # # #     """

# # # # # # # #     if not query or not query.strip():
# # # # # # # #         return {"status": "error", "message": "Query cannot be empty"}

# # # # # # # #     processed_query = resolve_query(query)
# # # # # # # #     logger.info(f"🔍 Search: original='{query}' expanded='{processed_query}'")

# # # # # # # #     # extract color words from query for color bonus
# # # # # # # #     COLOR_MAP = {
# # # # # # # #         'red': (1.0,0,0), 'blue': (0,0,1.0), 'green': (0,1.0,0),
# # # # # # # #         'yellow': (1.0,1.0,0), 'orange': (1.0,0.5,0), 'purple': (0.5,0,0.5),
# # # # # # # #         'pink': (1.0,0.75,0.8), 'black': (0,0,0), 'white': (1,1,1),
# # # # # # # #         'gray': (0.5,0.5,0.5), 'brown': (0.6,0.4,0.2)
# # # # # # # #     }
# # # # # # # #     query_lower = query.lower()
# # # # # # # #     query_colors = [rgb for name,rgb in COLOR_MAP.items() if name in query_lower]

# # # # # # # #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# # # # # # # #     if query_emb is None or search_engine.index is None:
# # # # # # # #         return {
# # # # # # # #             "status": "error",
# # # # # # # #             "message": "No images indexed yet. Please run build_index.py and upload photos first!"
# # # # # # # #         }

# # # # # # # #     # Search FAISS - focus on top matches only
# # # # # # # #     candidate_k = min(top_k * 8, 250)
# # # # # # # #     query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
# # # # # # # #     faiss.normalize_L2(query_emb_reshaped)
# # # # # # # #     distances, indices = search_engine.index.search(query_emb_reshaped, candidate_k)

# # # # # # # #     db = SessionLocal()
# # # # # # # #     results = []

# # # # # # # #     try:
# # # # # # # #         for dist, idx in zip(distances[0], indices[0]):
# # # # # # # #             if idx == -1:
# # # # # # # #                 continue
# # # # # # # #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# # # # # # # #             if not img:
# # # # # # # #                 logger.debug(f"Image with ID {idx} not found in database")
# # # # # # # #                 continue

# # # # # # # #             raw_sim = float(dist)
# # # # # # # #             clip_score = max(0.0, raw_sim)
            
# # # # # # # #             # CLIP score filtering: skip images with very low CLIP similarity
# # # # # # # #             if clip_score < CLIP_SCORE_MIN:
# # # # # # # #                 logger.debug(f"Skip {img.filename}: CLIP score {clip_score:.3f} below minimum {CLIP_SCORE_MIN}")
# # # # # # # #                 continue

# # # # # # # #             # ===== OCR MATCHING (Strong signal) =====
# # # # # # # #             ocr_text = (img.ocr_text or "").lower()
# # # # # # # #             query_words = processed_query.lower().split()
# # # # # # # #             significant_words = [w for w in query_words if len(w) > 2]
            
# # # # # # # #             ocr_bonus = 0.0
# # # # # # # #             if significant_words and ocr_text:
# # # # # # # #                 matches = sum(1 for w in significant_words if w in ocr_text)
# # # # # # # #                 if matches > 0:
# # # # # # # #                     ocr_bonus = min(matches / len(significant_words), 1.0)
# # # # # # # #                     logger.debug(f"{img.filename}: OCR matched {matches} words → bonus={ocr_bonus:.2f}")

# # # # # # # #             # ===== TAG MATCHING (Strong signal) =====
# # # # # # # #             tag_bonus = 0.0
# # # # # # # #             tags = []
# # # # # # # #             if img.scene_label:
# # # # # # # #                 tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
# # # # # # # #                 query_objects = [w for w in query_words if len(w) > 2]
# # # # # # # #                 if query_objects:
# # # # # # # #                     for obj in query_objects:
# # # # # # # #                         if any(obj in tag for tag in tags):
# # # # # # # #                             tag_bonus = 1.0
# # # # # # # #                             logger.debug(f"{img.filename}: TAG matched '{obj}' → bonus=1.0")
# # # # # # # #                             break

# # # # # # # #             # ===== COLOR BONUS =====
# # # # # # # #             color_bonus = 0.0
# # # # # # # #             if query_colors and getattr(img, 'avg_r', None) is not None:
# # # # # # # #                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# # # # # # # #                 for qc in query_colors:
# # # # # # # #                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# # # # # # # #                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
# # # # # # # #                     color_bonus = max(color_bonus, score)

# # # # # # # #             # ===== SMART COMBINED SCORING =====
# # # # # # # #             # If image has OCR or tag match, it's likely relevant → boost it!
# # # # # # # #             # If no match, need higher CLIP score to compensate
# # # # # # # #             if ocr_bonus > 0 or tag_bonus > 0:
# # # # # # # #                 # Image has keyword/tag confirmation - trust it more!
# # # # # # # #                 # This boosts images with matching keywords even if CLIP is weak
# # # # # # # #                 final_score = (
# # # # # # # #                     (0.40 * clip_score) +     # Reduce CLIP weight when confirmed
# # # # # # # #                     (0.35 * ocr_bonus) +      # OCR match is STRONG signal
# # # # # # # #                     (0.15 * color_bonus) +
# # # # # # # #                     (0.10 * tag_bonus)
# # # # # # # #                 )
# # # # # # # #                 logger.debug(f"{img.filename}: With OCR/Tag match → final={final_score:.3f}")
# # # # # # # #             else:
# # # # # # # #                 # No OCR/tag match - rely more on CLIP score alone
# # # # # # # #                 # This prevents random images from appearing
# # # # # # # #                 final_score = (
# # # # # # # #                     (0.70 * clip_score) +     # CLIP dominates when no confirmation
# # # # # # # #                     (0.15 * ocr_bonus) +
# # # # # # # #                     (0.10 * color_bonus) +
# # # # # # # #                     (0.05 * tag_bonus)
# # # # # # # #                 )
# # # # # # # #                 logger.debug(f"{img.filename}: No OCR/Tag match → final={final_score:.3f}")

# # # # # # # #             # ===== FILTER BY FINAL SCORE =====
# # # # # # # #             if final_score < FINAL_SCORE_MIN:
# # # # # # # #                 logger.debug(f"Skip {img.filename}: final_score {final_score:.3f} < {FINAL_SCORE_MIN}")
# # # # # # # #                 continue
            
# # # # # # # #             logger.debug(f"✅ MATCH: {img.filename}, CLIP={clip_score:.3f}, OCR={ocr_bonus:.2f}, TAG={tag_bonus:.2f}, FINAL={final_score:.3f}")

# # # # # # # #             results.append({
# # # # # # # #                 "id": img.id,
# # # # # # # #                 "filename": img.filename,
# # # # # # # #                 "score": round(final_score * 100, 2),
# # # # # # # #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # # # # # # #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # # # # # # #                 "person_count": img.person_count or 0
# # # # # # # #             })

# # # # # # # #         results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
# # # # # # # #         if not results:
# # # # # # # #             return {
# # # # # # # #                 "status": "not_found",
# # # # # # # #                 "message": f"No images found matching '{query}'",
# # # # # # # #                 "suggestion": "Try more specific keywords like 'dog', 'beach', 'sunset'"
# # # # # # # #             }

# # # # # # # #         logger.info(f"✅ Found {len(results)} results for '{query}'")
# # # # # # # #         return {
# # # # # # # #             "status": "found",
# # # # # # # #             "query": query,
# # # # # # # #             "count": len(results),
# # # # # # # #             "results": results
# # # # # # # #         }
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.post("/search/voice")
# # # # # # # # def voice_search(duration: int = Form(5)):
# # # # # # # #     """Search using voice input"""
# # # # # # # #     try:
# # # # # # # #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# # # # # # # #         if not transcribed:
# # # # # # # #             return {"status": "error", "message": "Could not transcribe audio"}
        
# # # # # # # #         # Perform search with transcribed text
# # # # # # # #         return search(query=transcribed, top_k=20)
# # # # # # # #     except Exception as e:
# # # # # # # #         logger.error(f"Voice search failed: {e}")
# # # # # # # #         return {"status": "error", "message": str(e)}

# # # # # # # # @app.get("/timeline")
# # # # # # # # def get_timeline():
# # # # # # # #     """Get all images organized chronologically"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
        
# # # # # # # #         results = []
# # # # # # # #         for img in images:
# # # # # # # #             results.append({
# # # # # # # #                 "id": img.id,
# # # # # # # #                 "filename": img.filename,
# # # # # # # #                 "date": img.timestamp.isoformat() if img.timestamp else None,
# # # # # # # #                 "thumbnail": f"/images/{img.filename}"
# # # # # # # #             })
        
# # # # # # # #         return {"count": len(results), "results": results}
# # # # # # # #     except Exception as e:
# # # # # # # #         logger.error(f"❌ Timeline error: {e}")
# # # # # # # #         raise HTTPException(status_code=500, detail=str(e))
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.get("/faces")
# # # # # # # # def get_faces(person_id: int = Query(None)):
# # # # # # # #     """Get detected people and their face counts"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         if person_id:
# # # # # # # #             # Get specific person's faces
# # # # # # # #             person = db.query(Person).filter(Person.id == person_id).first()
# # # # # # # #             if not person:
# # # # # # # #                 raise HTTPException(status_code=404, detail="Person not found")
            
# # # # # # # #             faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# # # # # # # #             images = []
# # # # # # # #             for face in faces:
# # # # # # # #                 img = db.query(DBImage).filter(DBImage.id == face.image_id).first()
# # # # # # # #                 if img:
# # # # # # # #                     images.append({
# # # # # # # #                         "id": img.id,
# # # # # # # #                         "filename": img.filename,
# # # # # # # #                         "thumbnail": f"/images/{img.filename}",
# # # # # # # #                         "date": img.timestamp.isoformat() if img.timestamp else None
# # # # # # # #                     })
            
# # # # # # # #             return {
# # # # # # # #                 "id": person.id,
# # # # # # # #                 "name": person.name,
# # # # # # # #                 "face_count": len(faces),
# # # # # # # #                 "images": images
# # # # # # # #             }
# # # # # # # #         else:
# # # # # # # #             # Get all people
# # # # # # # #             people = db.query(Person).all()
# # # # # # # #             results = []
# # # # # # # #             for p in people:
# # # # # # # #                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
                
# # # # # # # #                 if not faces:
# # # # # # # #                     continue
                
# # # # # # # #                 images = []
# # # # # # # #                 cover_filename = None
# # # # # # # #                 for face in faces:
# # # # # # # #                     img = db.query(DBImage).filter(DBImage.id == face.image_id).first()
# # # # # # # #                     if img:
# # # # # # # #                         if not cover_filename:
# # # # # # # #                             cover_filename = img.filename
# # # # # # # #                         images.append({
# # # # # # # #                             "id": img.id,
# # # # # # # #                             "filename": img.filename,
# # # # # # # #                             "thumbnail": f"/images/{img.filename}",
# # # # # # # #                             "date": img.timestamp.isoformat() if img.timestamp else None
# # # # # # # #                         })
                
# # # # # # # #                 results.append({
# # # # # # # #                     "id": p.id,
# # # # # # # #                     "name": p.name,
# # # # # # # #                     "count": len(faces),
# # # # # # # #                     "cover": f"/images/{cover_filename}" if cover_filename else None,
# # # # # # # #                     "images": images
# # # # # # # #                 })
            
# # # # # # # #             return {"results": results, "count": len(results)}
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.post("/people/{person_id}")
# # # # # # # # def update_person(person_id: int, name: str = Form(...)):
# # # # # # # #     """Rename a person"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # # # # # #         if not person:
# # # # # # # #             raise HTTPException(status_code=404, detail="Person not found")
        
# # # # # # # #         person.name = name
# # # # # # # #         db.commit()
# # # # # # # #         logger.info(f"✅ Renamed person {person_id} to '{name}'")
        
# # # # # # # #         return {
# # # # # # # #             "status": "success",
# # # # # # # #             "id": person.id,
# # # # # # # #             "name": person.name
# # # # # # # #         }
# # # # # # # #     except Exception as e:
# # # # # # # #         db.rollback()
# # # # # # # #         logger.error(f"Update person failed: {e}")
# # # # # # # #         raise HTTPException(status_code=500, detail=str(e))
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.get("/people/{person_id}/celebcheck")
# # # # # # # # def check_celebrity_match(person_id: int):
# # # # # # # #     """Try to identify if person matches a known celebrity"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # # # # # #         if not person:
# # # # # # # #             raise HTTPException(status_code=404, detail="Person not found")
        
# # # # # # # #         faces = db.query(DBFace).filter(DBFace.person_id == person_id).limit(1).all()
# # # # # # # #         if not faces or not faces[0].image_id:
# # # # # # # #             return {
# # # # # # # #                 "status": "no_match",
# # # # # # # #                 "message": "No face found for this person"
# # # # # # # #             }
        
# # # # # # # #         img = db.query(DBImage).filter(DBImage.id == faces[0].image_id).first()
# # # # # # # #         if not img or not os.path.exists(img.original_path):
# # # # # # # #             return {
# # # # # # # #                 "status": "no_match",
# # # # # # # #                 "message": "Face image not found"
# # # # # # # #             }
        
# # # # # # # #         ocr_text = img.ocr_text or ""
# # # # # # # #         if ocr_text:
# # # # # # # #             import re
# # # # # # # #             names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', ocr_text)
# # # # # # # #             if names:
# # # # # # # #                 logger.info(f"Suggested names from OCR: {names}")
# # # # # # # #                 return {
# # # # # # # #                     "status": "suggestions",
# # # # # # # #                     "suggestions": names[:3]
# # # # # # # #                 }
        
# # # # # # # #         return {
# # # # # # # #             "status": "no_match",
# # # # # # # #             "message": "Could not identify from available context. Try manual entry or Google Images.",
# # # # # # # #             "suggestion": "You can manually edit the name to identify this person"
# # # # # # # #         }
# # # # # # # #     except Exception as e:
# # # # # # # #         logger.error(f"Celebrity check failed: {e}")
# # # # # # # #         return {
# # # # # # # #             "status": "error",
# # # # # # # #             "message": str(e)
# # # # # # # #         }
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.get("/albums")
# # # # # # # # def get_albums(album_id: int = Query(None)):
# # # # # # # #     """Get auto-generated albums (trips/events)"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         if album_id:
# # # # # # # #             # Get specific album
# # # # # # # #             album = db.query(Album).filter(Album.id == album_id).first()
# # # # # # # #             if not album:
# # # # # # # #                 raise HTTPException(status_code=404, detail="Album not found")
            
# # # # # # # #             images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
            
# # # # # # # #             image_list = []
# # # # # # # #             for img in images:
# # # # # # # #                 image_list.append({
# # # # # # # #                     "id": img.id,
# # # # # # # #                     "filename": img.filename,
# # # # # # # #                     "date": img.timestamp.isoformat() if img.timestamp else None,
# # # # # # # #                     "thumbnail": f"/images/{img.filename}"
# # # # # # # #                 })
            
# # # # # # # #             return {
# # # # # # # #                 "id": album.id,
# # # # # # # #                 "title": album.title,
# # # # # # # #                 "type": album.type,
# # # # # # # #                 "description": album.description,
# # # # # # # #                 "start_date": album.start_date.isoformat() if album.start_date else None,
# # # # # # # #                 "end_date": album.end_date.isoformat() if album.end_date else None,
# # # # # # # #                 "image_count": len(images),
# # # # # # # #                 "images": image_list
# # # # # # # #             }
# # # # # # # #         else:
# # # # # # # #             # Get all albums
# # # # # # # #             albums = db.query(Album).all()
# # # # # # # #             results = []
            
# # # # # # # #             for a in albums:
# # # # # # # #                 album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
# # # # # # # #                 cover = f"/images/{album_images[0].filename}" if album_images else None
                
# # # # # # # #                 date_str = ""
# # # # # # # #                 if a.start_date:
# # # # # # # #                     date_str = a.start_date.strftime("%b %Y")
# # # # # # # #                     if a.end_date and a.end_date.month != a.start_date.month:
# # # # # # # #                         date_str += f" – {a.end_date.strftime('%b %Y')}"
                
# # # # # # # #                 results.append({
# # # # # # # #                     "id": a.id,
# # # # # # # #                     "title": a.title,
# # # # # # # #                     "type": a.type,
# # # # # # # #                     "cover": cover,
# # # # # # # #                     "count": len(album_images),
# # # # # # # #                     "date": date_str,
# # # # # # # #                     "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]]
# # # # # # # #                 })
            
# # # # # # # #             return {"results": results, "count": len(results)}
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.post("/favorites")
# # # # # # # # def add_favorite(image_id: int = Form(...)):
# # # # # # # #     """Mark image as favorite or toggle favorite status"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # # # # # # #         if not img:
# # # # # # # #             raise HTTPException(status_code=404, detail="Image not found")
        
# # # # # # # #         img.is_favorite = not getattr(img, 'is_favorite', False)
# # # # # # # #         db.commit()
        
# # # # # # # #         logger.info(f"Image {image_id} favorite toggled to {img.is_favorite}")
        
# # # # # # # #         return {
# # # # # # # #             "status": "success",
# # # # # # # #             "image_id": image_id,
# # # # # # # #             "is_favorite": img.is_favorite
# # # # # # # #         }
# # # # # # # #     except Exception as e:
# # # # # # # #         db.rollback()
# # # # # # # #         logger.error(f"Add favorite failed: {e}")
# # # # # # # #         raise HTTPException(status_code=500, detail=str(e))
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.get("/favorites")
# # # # # # # # def get_favorites():
# # # # # # # #     """Get all favorite images"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         images = db.query(DBImage).filter(DBImage.is_favorite == True).order_by(DBImage.timestamp.desc()).all()
        
# # # # # # # #         results = []
# # # # # # # #         for img in images:
# # # # # # # #             results.append({
# # # # # # # #                 "id": img.id,
# # # # # # # #                 "filename": img.filename,
# # # # # # # #                 "date": img.timestamp.isoformat() if img.timestamp else None,
# # # # # # # #                 "thumbnail": f"/images/{img.filename}"
# # # # # # # #             })
        
# # # # # # # #         logger.info(f"Retrieved {len(results)} favorite images")
# # # # # # # #         return {"count": len(results), "results": results}
# # # # # # # #     except Exception as e:
# # # # # # # #         logger.error(f"Get favorites error: {e}")
# # # # # # # #         return {"count": 0, "results": []}
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.get("/duplicates")
# # # # # # # # def get_duplicates():
# # # # # # # #     """Find and list duplicate images using perceptual hashing"""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         all_images = db.query(DBImage).all()
        
# # # # # # # #         duplicate_groups = []
# # # # # # # #         processed = set()
        
# # # # # # # #         logger.info(f"Scanning {len(all_images)} images for duplicates...")
        
# # # # # # # #         for i, img1 in enumerate(all_images):
# # # # # # # #             if img1.id in processed:
# # # # # # # #                 continue
            
# # # # # # # #             if not img1.original_path or not os.path.exists(img1.original_path):
# # # # # # # #                 continue
            
# # # # # # # #             duplicates_of_this = [img1]
            
# # # # # # # #             for img2 in all_images[i+1:]:
# # # # # # # #                 if img2.id in processed:
# # # # # # # #                     continue
                
# # # # # # # #                 if not img2.original_path or not os.path.exists(img2.original_path):
# # # # # # # #                     continue
                
# # # # # # # #                 hash1 = duplicate_engine.get_phash(img1.original_path)
# # # # # # # #                 hash2 = duplicate_engine.get_phash(img2.original_path)
                
# # # # # # # #                 if hash1 and hash2:
# # # # # # # #                     diff = bin(hash1 ^ hash2).count('1')
# # # # # # # #                     if diff < 5:
# # # # # # # #                         duplicates_of_this.append(img2)
# # # # # # # #                         processed.add(img2.id)
            
# # # # # # # #             if len(duplicates_of_this) > 1:
# # # # # # # #                 group = []
# # # # # # # #                 for img in duplicates_of_this:
# # # # # # # #                     size = os.path.getsize(img.original_path) if os.path.exists(img.original_path) else 0
# # # # # # # #                     group.append({
# # # # # # # #                         "id": img.id,
# # # # # # # #                         "filename": img.filename,
# # # # # # # #                         "thumbnail": f"/images/{img.filename}",
# # # # # # # #                         "size": size,
# # # # # # # #                         "date": img.timestamp.isoformat() if img.timestamp else None
# # # # # # # #                     })
                
# # # # # # # #                 duplicate_groups.append({
# # # # # # # #                     "count": len(group),
# # # # # # # #                     "images": group,
# # # # # # # #                     "total_size": sum(img['size'] for img in group)
# # # # # # # #                 })
                
# # # # # # # #                 for img in duplicates_of_this:
# # # # # # # #                     processed.add(img.id)
        
# # # # # # # #         logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        
# # # # # # # #         return {
# # # # # # # #             "status": "found" if duplicate_groups else "not_found",
# # # # # # # #             "duplicate_groups": duplicate_groups,
# # # # # # # #             "total_groups": len(duplicate_groups),
# # # # # # # #             "total_duplicates": sum(len(g["images"]) for g in duplicate_groups),
# # # # # # # #             "potential_savings_mb": sum(g["total_size"] for g in duplicate_groups) / (1024*1024)
# # # # # # # #         }
# # # # # # # #     except Exception as e:
# # # # # # # #         logger.error(f"Duplicate detection error: {e}")
# # # # # # # #         return {
# # # # # # # #             "status": "error",
# # # # # # # #             "message": str(e),
# # # # # # # #             "duplicate_groups": []
# # # # # # # #         }
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # @app.post("/recluster")
# # # # # # # # def recluster():
# # # # # # # #     """Clears old auto-generated people/albums and re-runs clustering from scratch."""
# # # # # # # #     db = SessionLocal()
# # # # # # # #     try:
# # # # # # # #         logger.info("🔄 Starting recluster operation...")
        
# # # # # # # #         # Clear old clustering data
# # # # # # # #         db.query(DBFace).update({"person_id": None})
# # # # # # # #         db.query(Person).delete()
# # # # # # # #         db.query(DBImage).update({"album_id": None})
# # # # # # # #         db.query(Album).filter(Album.type == "event").delete()
# # # # # # # #         db.commit()
# # # # # # # #         logger.info("✅ Cleared old clustering data")

# # # # # # # #         # Face clustering
# # # # # # # #         all_faces = db.query(DBFace).all()
# # # # # # # #         embeddings = []
# # # # # # # #         valid_faces = []
        
# # # # # # # #         for face in all_faces:
# # # # # # # #             if face.face_embedding is not None:
# # # # # # # #                 try:
# # # # # # # #                     arr = np.frombuffer(face.face_embedding, dtype=np.float32)
# # # # # # # #                     embeddings.append(arr)
# # # # # # # #                     valid_faces.append(face)
# # # # # # # #                 except Exception as e:
# # # # # # # #                     logger.warning(f"Could not parse embedding for face {face.id}: {e}")

# # # # # # # #         face_count = 0
# # # # # # # #         if embeddings:
# # # # # # # #             labels = face_engine.cluster_faces(embeddings)
# # # # # # # #             person_map = {}
            
# # # # # # # #             for i, label in enumerate(labels):
# # # # # # # #                 if label == -1:
# # # # # # # #                     continue
                
# # # # # # # #                 if label not in person_map:
# # # # # # # #                     new_person = Person(name=f"Person {label + 1}")
# # # # # # # #                     db.add(new_person)
# # # # # # # #                     db.flush()
# # # # # # # #                     person_map[label] = new_person.id
# # # # # # # #                     face_count += 1
                
# # # # # # # #                 valid_faces[i].person_id = person_map[label]
            
# # # # # # # #             db.commit()
# # # # # # # #             face_engine.rebuild_index(embeddings, [f.id for f in valid_faces])
# # # # # # # #             logger.info(f"✅ Clustered {len(embeddings)} faces into {face_count} people")
# # # # # # # #         else:
# # # # # # # #             logger.warning("⚠️  No face embeddings found")

# # # # # # # #         # Album/event clustering
# # # # # # # #         from clustering_engine import clustering_engine
# # # # # # # #         all_images = db.query(DBImage).all()
# # # # # # # #         album_count = 0
        
# # # # # # # #         if all_images:
# # # # # # # #             metadata = [
# # # # # # # #                 {
# # # # # # # #                     "id": img.id,
# # # # # # # #                     "lat": img.lat or 0.0,
# # # # # # # #                     "lon": img.lon or 0.0,
# # # # # # # #                     "timestamp": img.timestamp
# # # # # # # #                 }
# # # # # # # #                 for img in all_images if img.timestamp
# # # # # # # #             ]
            
# # # # # # # #             if metadata:
# # # # # # # #                 album_labels = clustering_engine.detect_events(metadata)
# # # # # # # #                 album_map = {}
                
# # # # # # # #                 for i, label in enumerate(album_labels):
# # # # # # # #                     if label == -1:
# # # # # # # #                         continue
                    
# # # # # # # #                     if label not in album_map:
# # # # # # # #                         cluster_imgs = [metadata[j] for j, l in enumerate(album_labels) if l == label]
# # # # # # # #                         ts_list = [m['timestamp'] for m in cluster_imgs if m['timestamp']]
                        
# # # # # # # #                         start_d = min(ts_list) if ts_list else None
# # # # # # # #                         end_d = max(ts_list) if ts_list else None
                        
# # # # # # # #                         new_album = Album(
# # # # # # # #                             title=f"Event {label + 1}",
# # # # # # # #                             type="event",
# # # # # # # #                             start_date=start_d,
# # # # # # # #                             end_date=end_d
# # # # # # # #                         )
# # # # # # # #                         db.add(new_album)
# # # # # # # #                         db.flush()
# # # # # # # #                         album_map[label] = new_album.id
# # # # # # # #                         album_count += 1
                    
# # # # # # # #                     all_images[i].album_id = album_map[label]
                
# # # # # # # #                 db.commit()
# # # # # # # #                 logger.info(f"✅ Created {album_count} albums")

# # # # # # # #         return {
# # # # # # # #             "status": "done",
# # # # # # # #             "people": face_count,
# # # # # # # #             "albums": album_count
# # # # # # # #         }
    
# # # # # # # #     except Exception as e:
# # # # # # # #         db.rollback()
# # # # # # # #         logger.error(f"❌ Recluster failed: {e}")
# # # # # # # #         raise HTTPException(status_code=500, detail=str(e))
# # # # # # # #     finally:
# # # # # # # #         db.close()


# # # # # # # # @app.post("/upload")
# # # # # # # # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# # # # # # # #     """Upload and process a single image"""
# # # # # # # #     ext = os.path.splitext(file.filename)[1].lower()
# # # # # # # #     if ext not in [".jpg", ".jpeg", ".png"]:
# # # # # # # #         raise HTTPException(status_code=400, detail="Unsupported file format. Use JPG or PNG.")
    
# # # # # # # #     filename = f"{uuid.uuid4()}{ext}"
# # # # # # # #     file_path = os.path.join(IMAGE_DIR, filename)
    
# # # # # # # #     db = SessionLocal()
    
# # # # # # # #     try:
# # # # # # # #         # Save File
# # # # # # # #         with open(file_path, "wb") as buffer:
# # # # # # # #             shutil.copyfileobj(file.file, buffer)
        
# # # # # # # #         logger.info(f"📤 Uploaded {filename}")
        
# # # # # # # #         # Get image dimensions
# # # # # # # #         from PIL import Image as PILImage
# # # # # # # #         try:
# # # # # # # #             img_pil = PILImage.open(file_path)
# # # # # # # #             width, height = img_pil.size
# # # # # # # #         except:
# # # # # # # #             width, height = None, None
        
# # # # # # # #         # compute average color
# # # # # # # #         try:
# # # # # # # #             im_color = img_pil.convert('RGB') if 'img_pil' in locals() else PILImage.open(file_path).convert('RGB')
# # # # # # # #             arr = np.array(im_color, dtype=np.float32)
# # # # # # # #             avg = arr.mean(axis=(0,1))
# # # # # # # #             avg_r, avg_g, avg_b = float(avg[0]), float(avg[1]), float(avg[2])
# # # # # # # #         except Exception:
# # # # # # # #             avg_r = avg_g = avg_b = 0.0
        
# # # # # # # #         # Semantic Embedding
# # # # # # # #         clip_emb = None
# # # # # # # #         try:
# # # # # # # #             clip_emb = search_engine.get_image_embedding(file_path)
# # # # # # # #             logger.info(f"✅ CLIP embedding extracted for {filename}")
# # # # # # # #         except Exception as e:
# # # # # # # #             logger.error(f"❌ CLIP embedding failed: {e}")

# # # # # # # #         # OCR
# # # # # # # #         ocr_text = ""
# # # # # # # #         try:
# # # # # # # #             ocr_text = extract_text(file_path)
# # # # # # # #             logger.info(f"✅ OCR completed: {len(ocr_text)} chars")
# # # # # # # #         except Exception as e:
# # # # # # # #             logger.error(f"❌ OCR failed: {e}")
        
# # # # # # # #         # Person Detection
# # # # # # # #         person_count = 0
# # # # # # # #         try:
# # # # # # # #             person_count = detector_engine.detect_persons(file_path)
# # # # # # # #             logger.info(f"✅ Detected {person_count} people")
# # # # # # # #         except Exception as e:
# # # # # # # #             logger.error(f"❌ Person detection failed: {e}")
        
# # # # # # # #         # Create Image Record
# # # # # # # #         img_record = DBImage(
# # # # # # # #             filename=filename,
# # # # # # # #             original_path=file_path,
# # # # # # # #             timestamp=datetime.now(),
# # # # # # # #             ocr_text=ocr_text,
# # # # # # # #             person_count=person_count,
# # # # # # # #             width=width,
# # # # # # # #             avg_r=avg_r,
# # # # # # # #             avg_g=avg_g,
# # # # # # # #             avg_b=avg_b,
# # # # # # # #             height=height,
# # # # # # # #             size_bytes=os.path.getsize(file_path)
# # # # # # # #         )
# # # # # # # #         db.add(img_record)
# # # # # # # #         db.flush()
        
# # # # # # # #         logger.info(f"✅ Created image record ID={img_record.id}")
        
# # # # # # # #         # Face Detection & Indexing
# # # # # # # #         face_count = 0
# # # # # # # #         try:
# # # # # # # #             faces = face_engine.detect_faces(file_path)
# # # # # # # #             logger.info(f"✅ Detected {len(faces)} faces")
            
# # # # # # # #             for face in faces:
# # # # # # # #                 emb = face['embedding'].astype(np.float32)
# # # # # # # #                 emb_blob = emb.tobytes()

# # # # # # # #                 # Try to match this face against existing face index
# # # # # # # #                 assigned_person_id = None
# # # # # # # #                 try:
# # # # # # # #                     if face_engine.face_index is not None and face_engine.face_index.ntotal > 0:
# # # # # # # #                         vec = emb.reshape(1, -1).astype('float32')
# # # # # # # #                         faiss.normalize_L2(vec)
# # # # # # # #                         D, I = face_engine.face_index.search(vec, FACE_MATCH_NEIGHBORS)
# # # # # # # #                         votes = {}
# # # # # # # #                         total_sim = 0.0
# # # # # # # #                         for sim_val, idx_pos in zip(D[0], I[0]):
# # # # # # # #                             if idx_pos == -1:
# # # # # # # #                                 continue
# # # # # # # #                             sim = float(sim_val)
# # # # # # # #                             total_sim += sim
# # # # # # # #                             try:
# # # # # # # #                                 matched_face_db_id = face_engine.face_id_map[idx_pos]
# # # # # # # #                                 matched_face = db.query(DBFace).filter(DBFace.id == int(matched_face_db_id)).first()
# # # # # # # #                                 if matched_face and matched_face.person_id:
# # # # # # # #                                     pid = matched_face.person_id
# # # # # # # #                                     votes[pid] = votes.get(pid, 0.0) + sim
# # # # # # # #                             except Exception:
# # # # # # # #                                 continue

# # # # # # # #                         if votes and total_sim > 0:
# # # # # # # #                             best_pid, best_score = max(votes.items(), key=lambda x: x[1])
# # # # # # # #                             if best_score / total_sim >= FACE_MATCH_VOTE_RATIO and best_score >= FACE_MATCH_THRESHOLD:
# # # # # # # #                                 assigned_person_id = best_pid
# # # # # # # #                 except Exception as e:
# # # # # # # #                     logger.warning(f"Face matching check failed: {e}")

# # # # # # # #                 # If no matching person found, create a new Person record
# # # # # # # #                 if assigned_person_id is None:
# # # # # # # #                     try:
# # # # # # # #                         new_person = Person(name=f"Person")
# # # # # # # #                         db.add(new_person)
# # # # # # # #                         db.flush()
# # # # # # # #                         assigned_person_id = new_person.id
# # # # # # # #                     except Exception as e:
# # # # # # # #                         logger.error(f"Could not create person record: {e}")
# # # # # # # #                         assigned_person_id = None

# # # # # # # #                 face_record = DBFace(
# # # # # # # #                     image_id=img_record.id,
# # # # # # # #                     bbox=str(face['bbox']),
# # # # # # # #                     face_embedding=emb_blob,
# # # # # # # #                     person_id=assigned_person_id
# # # # # # # #                 )
# # # # # # # #                 db.add(face_record)
# # # # # # # #                 db.flush()

# # # # # # # #                 # Add to face FAISS index and maintain id map
# # # # # # # #                 try:
# # # # # # # #                     face_engine.add_to_index(emb, face_record.id)
# # # # # # # #                 except Exception as e:
# # # # # # # #                     logger.error(f"Failed to add face to index: {e}")

# # # # # # # #                 face_count += 1
# # # # # # # #                 logger.info(f"✅ Processed and indexed face {face_count} (person_id={assigned_person_id})")
        
# # # # # # # #         except Exception as e:
# # # # # # # #             logger.error(f"❌ Face detection failed: {e}")
        
# # # # # # # #         # Update Image FAISS Index
# # # # # # # #         if clip_emb is not None:
# # # # # # # #             try:
# # # # # # # #                 if search_engine.index is None:
# # # # # # # #                     dim = clip_emb.shape[0]
# # # # # # # #                     sub_index = faiss.IndexFlatIP(dim)
# # # # # # # #                     search_engine.index = faiss.IndexIDMap(sub_index)
                
# # # # # # # #                 new_vec = clip_emb.reshape(1, -1).astype('float32')
# # # # # # # #                 faiss.normalize_L2(new_vec)
                
# # # # # # # #                 ids_np = np.array([img_record.id]).astype('int64')
# # # # # # # #                 search_engine.index.add_with_ids(new_vec, ids_np)
                
# # # # # # # #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# # # # # # # #                 logger.info(f"✅ Updated image FAISS index")
# # # # # # # #             except Exception as e:
# # # # # # # #                 logger.error(f"❌ Index update failed: {e}")
        
# # # # # # # #         db.commit()

# # # # # # # #         # Batched/debounced recluster
# # # # # # # #         if background_tasks is not None:
# # # # # # # #             should_trigger_recluster(background_tasks)

# # # # # # # #         return {
# # # # # # # #             "status": "success",
# # # # # # # #             "id": img_record.id,
# # # # # # # #             "filename": filename,
# # # # # # # #             "person_count": person_count,
# # # # # # # #             "face_count": face_count
# # # # # # # #         }
    
# # # # # # # #     except Exception as e:
# # # # # # # #         db.rollback()
# # # # # # # #         if os.path.exists(file_path):
# # # # # # # #             os.remove(file_path)
# # # # # # # #         logger.error(f"❌ Upload failed: {e}")
# # # # # # # #         raise HTTPException(status_code=500, detail=str(e))
    
# # # # # # # #     finally:
# # # # # # # #         db.close()

# # # # # # # # if __name__ == "__main__":
# # # # # # # #     import uvicorn
# # # # # # # #     uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# # # # # # # # FIXED main.py - Ultra-Strict Two-Tier Search Filtering
# # # # # # # # Images WITHOUT keyword matches are REJECTED unless CLIP > 0.22
# # # # # # # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# # # # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # # # from fastapi.staticfiles import StaticFiles
# # # # # # # import os
# # # # # # # import uuid
# # # # # # # import shutil
# # # # # # # import numpy as np
# # # # # # # import faiss
# # # # # # # from datetime import datetime
# # # # # # # import logging
# # # # # # # from contextlib import asynccontextmanager
# # # # # # # import datetime as datetime_module

# # # # # # # # Setup Logging
# # # # # # # logging.basicConfig(level=logging.INFO)
# # # # # # # logger = logging.getLogger("main")

# # # # # # # from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
# # # # # # # from search_engine import search_engine, resolve_query
# # # # # # # from voice_engine import voice_engine
# # # # # # # from face_engine import face_engine
# # # # # # # from ocr_engine import extract_text
# # # # # # # from detector_engine import detector_engine
# # # # # # # from duplicate_engine import duplicate_engine

# # # # # # # # Paths
# # # # # # # IMAGE_DIR = "../data/images"
# # # # # # # FAISS_INDEX_PATH = "../data/index.faiss"

# # # # # # # # Configurable thresholds and options
# # # # # # # FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
# # # # # # # FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
# # # # # # # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # # # # # # RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # # # # # # RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # # # # # # # Search thresholds - ULTRA-STRICT TWO-TIER SYSTEM
# # # # # # # CLIP_SCORE_MIN = 0.10      # Very low - let combined score decide
# # # # # # # FINAL_SCORE_MIN = 0.15     # With keyword confirmation
# # # # # # # STRICT_CLIP_MIN = 0.22     # Without keyword confirmation (very strict!)
# # # # # # # SEARCH_SCORE_THRESHOLD = float(os.environ.get("SEARCH_SCORE_THRESHOLD", 0.20))

# # # # # # # RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # # # # # # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # # # # # # recluster_last_triggered = None

# # # # # # # def should_trigger_recluster(background_tasks):
# # # # # # #     """Check if batch size reached or timer expired; schedule recluster if needed."""
# # # # # # #     global recluster_last_triggered
    
# # # # # # #     if not RECLUSTER_ON_UPLOAD or not background_tasks:
# # # # # # #         return
    
# # # # # # #     try:
# # # # # # #         counter = 0
# # # # # # #         if os.path.exists(RECLUSTER_COUNTER_PATH):
# # # # # # #             try:
# # # # # # #                 with open(RECLUSTER_COUNTER_PATH, 'r') as f:
# # # # # # #                     counter = int(f.read().strip())
# # # # # # #             except:
# # # # # # #                 pass
        
# # # # # # #         counter += 1
# # # # # # #         with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # # # # #             f.write(str(counter))
        
# # # # # # #         should_trigger = counter >= RECLUSTER_BATCH_SIZE
        
# # # # # # #         now = datetime_module.datetime.now()
# # # # # # #         if recluster_last_triggered:
# # # # # # #             elapsed = (now - recluster_last_triggered).total_seconds()
# # # # # # #             if elapsed >= RECLUSTER_TIMER_SECONDS:
# # # # # # #                 should_trigger = True
# # # # # # #         elif counter > 0:
# # # # # # #             should_trigger = counter >= RECLUSTER_BATCH_SIZE
        
# # # # # # #         if should_trigger:
# # # # # # #             logger.info(f"📊 Recluster triggered: counter={counter}, batch_size={RECLUSTER_BATCH_SIZE}")
# # # # # # #             background_tasks.add_task(recluster)
# # # # # # #             recluster_last_triggered = now
# # # # # # #             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # # # # #                 f.write('0')
# # # # # # #     except Exception as e:
# # # # # # #         logger.warning(f"Batched recluster check failed: {e}")

# # # # # # # @asynccontextmanager
# # # # # # # async def lifespan(app: FastAPI):
# # # # # # #     init_db()
# # # # # # #     logger.info("🚀 Starting Offline Smart Gallery Backend...")

# # # # # # #     if os.path.exists(FAISS_INDEX_PATH):
# # # # # # #         try:
# # # # # # #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# # # # # # #             logger.info(f"✅ Image FAISS index loaded ({search_engine.index.ntotal} vectors).")
# # # # # # #         except Exception as e:
# # # # # # #             logger.error(f"❌ Error loading image FAISS index: {e}")
# # # # # # #             search_engine.index = None
# # # # # # #     else:
# # # # # # #         logger.warning("⚠️  Image FAISS index not found. Searching will be limited.")

# # # # # # #     logger.info(f"✅ Face FAISS index ready ({face_engine.face_index.ntotal if face_engine.face_index else 0} vectors).")
# # # # # # #     yield

# # # # # # # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)

# # # # # # # app.add_middleware(
# # # # # # #     CORSMiddleware,
# # # # # # #     allow_origins=["*"],
# # # # # # #     allow_methods=["*"],
# # # # # # #     allow_headers=["*"],
# # # # # # # )

# # # # # # # if not os.path.exists(IMAGE_DIR):
# # # # # # #     os.makedirs(IMAGE_DIR)

# # # # # # # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # # # # # # @app.get("/health")
# # # # # # # def health():
# # # # # # #     return {
# # # # # # #         "status": "ready",
# # # # # # #         "models": ["CLIP", "OCR", "FaceRecognition", "Clustering"],
# # # # # # #         "image_index": search_engine.index.ntotal if search_engine.index else 0,
# # # # # # #         "face_index": face_engine.face_index.ntotal if face_engine.face_index else 0
# # # # # # #     }

# # # # # # # @app.get("/test-db")
# # # # # # # def test_db():
# # # # # # #     """Diagnostic endpoint to test database"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         count = db.query(DBImage).count()
# # # # # # #         images = db.query(DBImage).limit(1).all()
# # # # # # #         return {
# # # # # # #             "status": "ok",
# # # # # # #             "total_images": count,
# # # # # # #             "sample": {
# # # # # # #                 "filename": images[0].filename,
# # # # # # #                 "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None
# # # # # # #             } if images else None
# # # # # # #         }
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"❌ Test DB error: {e}", exc_info=True)
# # # # # # #         return {"status": "error", "message": str(e)}
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.post("/search")
# # # # # # # def search(query: str = Form(...), top_k: int = Form(20)):
# # # # # # #     """
# # # # # # #     ULTRA-STRICT TWO-TIER SEARCH:
# # # # # # #     - Images WITH matching keywords: Can have lower CLIP (0.15+ threshold)
# # # # # # #     - Images WITHOUT keywords: Need high CLIP (0.22+ threshold)
    
# # # # # # #     This eliminates random unrelated images while allowing confirmed matches!
# # # # # # #     """

# # # # # # #     if not query or not query.strip():
# # # # # # #         return {"status": "error", "message": "Query cannot be empty"}

# # # # # # #     processed_query = resolve_query(query)
# # # # # # #     logger.info(f"🔍 Search: original='{query}' expanded='{processed_query}'")

# # # # # # #     # Extract color words from query for color bonus
# # # # # # #     COLOR_MAP = {
# # # # # # #         'red': (1.0, 0, 0), 'blue': (0, 0, 1.0), 'green': (0, 1.0, 0),
# # # # # # #         'yellow': (1.0, 1.0, 0), 'orange': (1.0, 0.5, 0), 'purple': (0.5, 0, 0.5),
# # # # # # #         'pink': (1.0, 0.75, 0.8), 'black': (0, 0, 0), 'white': (1, 1, 1),
# # # # # # #         'gray': (0.5, 0.5, 0.5), 'brown': (0.6, 0.4, 0.2)
# # # # # # #     }
# # # # # # #     query_lower = query.lower()
# # # # # # #     query_colors = [rgb for name, rgb in COLOR_MAP.items() if name in query_lower]

# # # # # # #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# # # # # # #     if query_emb is None or search_engine.index is None:
# # # # # # #         return {
# # # # # # #             "status": "error",
# # # # # # #             "message": "No images indexed yet. Please run build_index.py and upload photos first!"
# # # # # # #         }

# # # # # # #     # Search FAISS - focus on top matches only
# # # # # # #     candidate_k = min(top_k * 8, 250)
# # # # # # #     query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
# # # # # # #     faiss.normalize_L2(query_emb_reshaped)
# # # # # # #     distances, indices = search_engine.index.search(query_emb_reshaped, candidate_k)

# # # # # # #     db = SessionLocal()
# # # # # # #     results = []

# # # # # # #     try:
# # # # # # #         for dist, idx in zip(distances[0], indices[0]):
# # # # # # #             if idx == -1:
# # # # # # #                 continue
# # # # # # #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# # # # # # #             if not img:
# # # # # # #                 logger.debug(f"Image with ID {idx} not found in database")
# # # # # # #                 continue

# # # # # # #             raw_sim = float(dist)
# # # # # # #             clip_score = max(0.0, raw_sim)
            
# # # # # # #             # CLIP score filtering: skip images with very low CLIP similarity
# # # # # # #             if clip_score < CLIP_SCORE_MIN:
# # # # # # #                 logger.debug(f"Skip {img.filename}: CLIP score {clip_score:.3f} below minimum {CLIP_SCORE_MIN}")
# # # # # # #                 continue

# # # # # # #             # ===== OCR MATCHING =====
# # # # # # #             ocr_text = (img.ocr_text or "").lower()
# # # # # # #             query_words = processed_query.lower().split()
# # # # # # #             significant_words = [w for w in query_words if len(w) > 2]
            
# # # # # # #             ocr_bonus = 0.0
# # # # # # #             if significant_words and ocr_text:
# # # # # # #                 matches = sum(1 for w in significant_words if w in ocr_text)
# # # # # # #                 if matches > 0:
# # # # # # #                     ocr_bonus = min(matches / len(significant_words), 1.0)
# # # # # # #                     logger.debug(f"{img.filename}: OCR matched {matches} words → bonus={ocr_bonus:.2f}")

# # # # # # #             # ===== TAG MATCHING =====
# # # # # # #             tag_bonus = 0.0
# # # # # # #             tags = []
# # # # # # #             if img.scene_label:
# # # # # # #                 tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
# # # # # # #                 query_objects = [w for w in query_words if len(w) > 2]
# # # # # # #                 if query_objects:
# # # # # # #                     for obj in query_objects:
# # # # # # #                         if any(obj in tag for tag in tags):
# # # # # # #                             tag_bonus = 1.0
# # # # # # #                             logger.debug(f"{img.filename}: TAG matched '{obj}' → bonus=1.0")
# # # # # # #                             break

# # # # # # #             # ===== COLOR BONUS =====
# # # # # # #             color_bonus = 0.0
# # # # # # #             if query_colors and getattr(img, 'avg_r', None) is not None:
# # # # # # #                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# # # # # # #                 for qc in query_colors:
# # # # # # #                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# # # # # # #                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
# # # # # # #                     color_bonus = max(color_bonus, score)

# # # # # # #             # ===== ULTRA-STRICT TWO-TIER SCORING =====
# # # # # # #             # Decision: Does image have matching keywords/objects?
# # # # # # #             has_keyword_match = ocr_bonus > 0 or tag_bonus > 0

# # # # # # #             if has_keyword_match:
# # # # # # #                 # TIER 1: Image HAS matching keywords!
# # # # # # #                 # Lower threshold (0.15) because we've confirmed it's relevant
# # # # # # #                 final_score = (
# # # # # # #                     (0.40 * clip_score) +     # Reduce CLIP weight (confirmed)
# # # # # # #                     (0.35 * ocr_bonus) +      # OCR/tag match is PRIMARY signal
# # # # # # #                     (0.15 * color_bonus) +
# # # # # # #                     (0.10 * tag_bonus)
# # # # # # #                 )
# # # # # # #                 min_score_threshold = FINAL_SCORE_MIN  # 0.15 - lenient
# # # # # # #                 match_type = "KEYWORD_CONFIRMED"
                
# # # # # # #             else:
# # # # # # #                 # TIER 2: Image has NO matching keywords!
# # # # # # #                 # Use ONLY CLIP score, very strict threshold (0.22+)
# # # # # # #                 # This rejects "cat" when searching for "girl"
# # # # # # #                 final_score = clip_score
# # # # # # #                 min_score_threshold = STRICT_CLIP_MIN  # 0.22 - very strict
# # # # # # #                 match_type = "CLIP_ONLY"

# # # # # # #             # ===== APPLY THRESHOLD =====
# # # # # # #             if final_score < min_score_threshold:
# # # # # # #                 logger.debug(f"❌ Skip {img.filename}: {match_type} score {final_score:.3f} < {min_score_threshold} (CLIP={clip_score:.3f})")
# # # # # # #                 continue
            
# # # # # # #             logger.debug(f"✅ MATCH: {img.filename} ({match_type}), CLIP={clip_score:.3f}, OCR={ocr_bonus:.2f}, TAG={tag_bonus:.2f}, FINAL={final_score:.3f}")

# # # # # # #             results.append({
# # # # # # #                 "id": img.id,
# # # # # # #                 "filename": img.filename,
# # # # # # #                 "score": round(final_score * 100, 2),
# # # # # # #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # # # # # #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # # # # # #                 "person_count": img.person_count or 0
# # # # # # #             })

# # # # # # #         results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
# # # # # # #         if not results:
# # # # # # #             return {
# # # # # # #                 "status": "not_found",
# # # # # # #                 "message": f"No images found matching '{query}'",
# # # # # # #                 "suggestion": "Try more specific keywords like 'dog', 'beach', 'sunset'"
# # # # # # #             }

# # # # # # #         logger.info(f"✅ Found {len(results)} results for '{query}'")
# # # # # # #         return {
# # # # # # #             "status": "found",
# # # # # # #             "query": query,
# # # # # # #             "count": len(results),
# # # # # # #             "results": results
# # # # # # #         }
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.post("/search/voice")
# # # # # # # def voice_search(duration: int = Form(5)):
# # # # # # #     """Search using voice input"""
# # # # # # #     try:
# # # # # # #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# # # # # # #         if not transcribed:
# # # # # # #             return {"status": "error", "message": "Could not transcribe audio"}
        
# # # # # # #         return search(query=transcribed, top_k=20)
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"Voice search failed: {e}")
# # # # # # #         return {"status": "error", "message": str(e)}

# # # # # # # @app.get("/timeline")
# # # # # # # def get_timeline():
# # # # # # #     """Get all images organized chronologically"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
        
# # # # # # #         results = []
# # # # # # #         for img in images:
# # # # # # #             results.append({
# # # # # # #                 "id": img.id,
# # # # # # #                 "filename": img.filename,
# # # # # # #                 "date": img.timestamp.isoformat() if img.timestamp else None,
# # # # # # #                 "thumbnail": f"/images/{img.filename}"
# # # # # # #             })
        
# # # # # # #         return {"count": len(results), "results": results}
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"❌ Timeline error: {e}")
# # # # # # #         raise HTTPException(status_code=500, detail=str(e))
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.get("/faces")
# # # # # # # def get_faces(person_id: int = Query(None)):
# # # # # # #     """Get detected people and their face counts"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         if person_id:
# # # # # # #             person = db.query(Person).filter(Person.id == person_id).first()
# # # # # # #             if not person:
# # # # # # #                 raise HTTPException(status_code=404, detail="Person not found")
            
# # # # # # #             faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# # # # # # #             images = []
# # # # # # #             for face in faces:
# # # # # # #                 img = db.query(DBImage).filter(DBImage.id == face.image_id).first()
# # # # # # #                 if img:
# # # # # # #                     images.append({
# # # # # # #                         "id": img.id,
# # # # # # #                         "filename": img.filename,
# # # # # # #                         "thumbnail": f"/images/{img.filename}",
# # # # # # #                         "date": img.timestamp.isoformat() if img.timestamp else None
# # # # # # #                     })
            
# # # # # # #             return {
# # # # # # #                 "id": person.id,
# # # # # # #                 "name": person.name,
# # # # # # #                 "face_count": len(faces),
# # # # # # #                 "images": images
# # # # # # #             }
# # # # # # #         else:
# # # # # # #             people = db.query(Person).all()
# # # # # # #             results = []
# # # # # # #             for p in people:
# # # # # # #                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
                
# # # # # # #                 if not faces:
# # # # # # #                     continue
                
# # # # # # #                 images = []
# # # # # # #                 cover_filename = None
# # # # # # #                 for face in faces:
# # # # # # #                     img = db.query(DBImage).filter(DBImage.id == face.image_id).first()
# # # # # # #                     if img:
# # # # # # #                         if not cover_filename:
# # # # # # #                             cover_filename = img.filename
# # # # # # #                         images.append({
# # # # # # #                             "id": img.id,
# # # # # # #                             "filename": img.filename,
# # # # # # #                             "thumbnail": f"/images/{img.filename}",
# # # # # # #                             "date": img.timestamp.isoformat() if img.timestamp else None
# # # # # # #                         })
                
# # # # # # #                 results.append({
# # # # # # #                     "id": p.id,
# # # # # # #                     "name": p.name,
# # # # # # #                     "count": len(faces),
# # # # # # #                     "cover": f"/images/{cover_filename}" if cover_filename else None,
# # # # # # #                     "images": images
# # # # # # #                 })
            
# # # # # # #             return {"results": results, "count": len(results)}
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.post("/people/{person_id}")
# # # # # # # def update_person(person_id: int, name: str = Form(...)):
# # # # # # #     """Rename a person"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # # # # #         if not person:
# # # # # # #             raise HTTPException(status_code=404, detail="Person not found")
        
# # # # # # #         person.name = name
# # # # # # #         db.commit()
# # # # # # #         logger.info(f"✅ Renamed person {person_id} to '{name}'")
        
# # # # # # #         return {
# # # # # # #             "status": "success",
# # # # # # #             "id": person.id,
# # # # # # #             "name": person.name
# # # # # # #         }
# # # # # # #     except Exception as e:
# # # # # # #         db.rollback()
# # # # # # #         logger.error(f"Update person failed: {e}")
# # # # # # #         raise HTTPException(status_code=500, detail=str(e))
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.get("/people/{person_id}/celebcheck")
# # # # # # # def check_celebrity_match(person_id: int):
# # # # # # #     """Try to identify if person matches a known celebrity"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # # # # #         if not person:
# # # # # # #             raise HTTPException(status_code=404, detail="Person not found")
        
# # # # # # #         faces = db.query(DBFace).filter(DBFace.person_id == person_id).limit(1).all()
# # # # # # #         if not faces or not faces[0].image_id:
# # # # # # #             return {
# # # # # # #                 "status": "no_match",
# # # # # # #                 "message": "No face found for this person"
# # # # # # #             }
        
# # # # # # #         img = db.query(DBImage).filter(DBImage.id == faces[0].image_id).first()
# # # # # # #         if not img or not os.path.exists(img.original_path):
# # # # # # #             return {
# # # # # # #                 "status": "no_match",
# # # # # # #                 "message": "Face image not found"
# # # # # # #             }
        
# # # # # # #         ocr_text = img.ocr_text or ""
# # # # # # #         if ocr_text:
# # # # # # #             import re
# # # # # # #             names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', ocr_text)
# # # # # # #             if names:
# # # # # # #                 logger.info(f"Suggested names from OCR: {names}")
# # # # # # #                 return {
# # # # # # #                     "status": "suggestions",
# # # # # # #                     "suggestions": names[:3]
# # # # # # #                 }
        
# # # # # # #         return {
# # # # # # #             "status": "no_match",
# # # # # # #             "message": "Could not identify from available context. Try manual entry or Google Images.",
# # # # # # #             "suggestion": "You can manually edit the name to identify this person"
# # # # # # #         }
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"Celebrity check failed: {e}")
# # # # # # #         return {
# # # # # # #             "status": "error",
# # # # # # #             "message": str(e)
# # # # # # #         }
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.get("/albums")
# # # # # # # def get_albums(album_id: int = Query(None)):
# # # # # # #     """Get auto-generated albums (trips/events)"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         if album_id:
# # # # # # #             album = db.query(Album).filter(Album.id == album_id).first()
# # # # # # #             if not album:
# # # # # # #                 raise HTTPException(status_code=404, detail="Album not found")
            
# # # # # # #             images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
            
# # # # # # #             image_list = []
# # # # # # #             for img in images:
# # # # # # #                 image_list.append({
# # # # # # #                     "id": img.id,
# # # # # # #                     "filename": img.filename,
# # # # # # #                     "date": img.timestamp.isoformat() if img.timestamp else None,
# # # # # # #                     "thumbnail": f"/images/{img.filename}"
# # # # # # #                 })
            
# # # # # # #             return {
# # # # # # #                 "id": album.id,
# # # # # # #                 "title": album.title,
# # # # # # #                 "type": album.type,
# # # # # # #                 "description": album.description,
# # # # # # #                 "start_date": album.start_date.isoformat() if album.start_date else None,
# # # # # # #                 "end_date": album.end_date.isoformat() if album.end_date else None,
# # # # # # #                 "image_count": len(images),
# # # # # # #                 "images": image_list
# # # # # # #             }
# # # # # # #         else:
# # # # # # #             albums = db.query(Album).all()
# # # # # # #             results = []
            
# # # # # # #             for a in albums:
# # # # # # #                 album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
# # # # # # #                 cover = f"/images/{album_images[0].filename}" if album_images else None
                
# # # # # # #                 date_str = ""
# # # # # # #                 if a.start_date:
# # # # # # #                     date_str = a.start_date.strftime("%b %Y")
# # # # # # #                     if a.end_date and a.end_date.month != a.start_date.month:
# # # # # # #                         date_str += f" – {a.end_date.strftime('%b %Y')}"
                
# # # # # # #                 results.append({
# # # # # # #                     "id": a.id,
# # # # # # #                     "title": a.title,
# # # # # # #                     "type": a.type,
# # # # # # #                     "cover": cover,
# # # # # # #                     "count": len(album_images),
# # # # # # #                     "date": date_str,
# # # # # # #                     "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]]
# # # # # # #                 })
            
# # # # # # #             return {"results": results, "count": len(results)}
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.post("/favorites")
# # # # # # # def add_favorite(image_id: int = Form(...)):
# # # # # # #     """Mark image as favorite or toggle favorite status"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # # # # # #         if not img:
# # # # # # #             raise HTTPException(status_code=404, detail="Image not found")
        
# # # # # # #         img.is_favorite = not getattr(img, 'is_favorite', False)
# # # # # # #         db.commit()
        
# # # # # # #         logger.info(f"Image {image_id} favorite toggled to {img.is_favorite}")
        
# # # # # # #         return {
# # # # # # #             "status": "success",
# # # # # # #             "image_id": image_id,
# # # # # # #             "is_favorite": img.is_favorite
# # # # # # #         }
# # # # # # #     except Exception as e:
# # # # # # #         db.rollback()
# # # # # # #         logger.error(f"Add favorite failed: {e}")
# # # # # # #         raise HTTPException(status_code=500, detail=str(e))
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.get("/favorites")
# # # # # # # def get_favorites():
# # # # # # #     """Get all favorite images"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         images = db.query(DBImage).filter(DBImage.is_favorite == True).order_by(DBImage.timestamp.desc()).all()
        
# # # # # # #         results = []
# # # # # # #         for img in images:
# # # # # # #             results.append({
# # # # # # #                 "id": img.id,
# # # # # # #                 "filename": img.filename,
# # # # # # #                 "date": img.timestamp.isoformat() if img.timestamp else None,
# # # # # # #                 "thumbnail": f"/images/{img.filename}"
# # # # # # #             })
        
# # # # # # #         logger.info(f"Retrieved {len(results)} favorite images")
# # # # # # #         return {"count": len(results), "results": results}
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"Get favorites error: {e}")
# # # # # # #         return {"count": 0, "results": []}
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.get("/duplicates")
# # # # # # # def get_duplicates():
# # # # # # #     """Find and list duplicate images using perceptual hashing"""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         all_images = db.query(DBImage).all()
        
# # # # # # #         duplicate_groups = []
# # # # # # #         processed = set()
        
# # # # # # #         logger.info(f"Scanning {len(all_images)} images for duplicates...")
        
# # # # # # #         for i, img1 in enumerate(all_images):
# # # # # # #             if img1.id in processed:
# # # # # # #                 continue
            
# # # # # # #             if not img1.original_path or not os.path.exists(img1.original_path):
# # # # # # #                 continue
            
# # # # # # #             duplicates_of_this = [img1]
            
# # # # # # #             for img2 in all_images[i+1:]:
# # # # # # #                 if img2.id in processed:
# # # # # # #                     continue
                
# # # # # # #                 if not img2.original_path or not os.path.exists(img2.original_path):
# # # # # # #                     continue
                
# # # # # # #                 hash1 = duplicate_engine.get_phash(img1.original_path)
# # # # # # #                 hash2 = duplicate_engine.get_phash(img2.original_path)
                
# # # # # # #                 if hash1 and hash2:
# # # # # # #                     diff = bin(hash1 ^ hash2).count('1')
# # # # # # #                     if diff < 5:
# # # # # # #                         duplicates_of_this.append(img2)
# # # # # # #                         processed.add(img2.id)
            
# # # # # # #             if len(duplicates_of_this) > 1:
# # # # # # #                 group = []
# # # # # # #                 for img in duplicates_of_this:
# # # # # # #                     size = os.path.getsize(img.original_path) if os.path.exists(img.original_path) else 0
# # # # # # #                     group.append({
# # # # # # #                         "id": img.id,
# # # # # # #                         "filename": img.filename,
# # # # # # #                         "thumbnail": f"/images/{img.filename}",
# # # # # # #                         "size": size,
# # # # # # #                         "date": img.timestamp.isoformat() if img.timestamp else None
# # # # # # #                     })
                
# # # # # # #                 duplicate_groups.append({
# # # # # # #                     "count": len(group),
# # # # # # #                     "images": group,
# # # # # # #                     "total_size": sum(img['size'] for img in group)
# # # # # # #                 })
                
# # # # # # #                 for img in duplicates_of_this:
# # # # # # #                     processed.add(img.id)
        
# # # # # # #         logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        
# # # # # # #         return {
# # # # # # #             "status": "found" if duplicate_groups else "not_found",
# # # # # # #             "duplicate_groups": duplicate_groups,
# # # # # # #             "total_groups": len(duplicate_groups),
# # # # # # #             "total_duplicates": sum(len(g["images"]) for g in duplicate_groups),
# # # # # # #             "potential_savings_mb": sum(g["total_size"] for g in duplicate_groups) / (1024*1024)
# # # # # # #         }
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"Duplicate detection error: {e}")
# # # # # # #         return {
# # # # # # #             "status": "error",
# # # # # # #             "message": str(e),
# # # # # # #             "duplicate_groups": []
# # # # # # #         }
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.post("/recluster")
# # # # # # # def recluster():
# # # # # # #     """Clears old auto-generated people/albums and re-runs clustering from scratch."""
# # # # # # #     db = SessionLocal()
# # # # # # #     try:
# # # # # # #         logger.info("🔄 Starting recluster operation...")
        
# # # # # # #         db.query(DBFace).update({"person_id": None})
# # # # # # #         db.query(Person).delete()
# # # # # # #         db.query(DBImage).update({"album_id": None})
# # # # # # #         db.query(Album).filter(Album.type == "event").delete()
# # # # # # #         db.commit()
# # # # # # #         logger.info("✅ Cleared old clustering data")

# # # # # # #         all_faces = db.query(DBFace).all()
# # # # # # #         embeddings = []
# # # # # # #         valid_faces = []
        
# # # # # # #         for face in all_faces:
# # # # # # #             if face.face_embedding is not None:
# # # # # # #                 try:
# # # # # # #                     arr = np.frombuffer(face.face_embedding, dtype=np.float32)
# # # # # # #                     embeddings.append(arr)
# # # # # # #                     valid_faces.append(face)
# # # # # # #                 except Exception as e:
# # # # # # #                     logger.warning(f"Could not parse embedding for face {face.id}: {e}")

# # # # # # #         face_count = 0
# # # # # # #         if embeddings:
# # # # # # #             labels = face_engine.cluster_faces(embeddings)
# # # # # # #             person_map = {}
            
# # # # # # #             for i, label in enumerate(labels):
# # # # # # #                 if label == -1:
# # # # # # #                     continue
                
# # # # # # #                 if label not in person_map:
# # # # # # #                     new_person = Person(name=f"Person {label + 1}")
# # # # # # #                     db.add(new_person)
# # # # # # #                     db.flush()
# # # # # # #                     person_map[label] = new_person.id
# # # # # # #                     face_count += 1
                
# # # # # # #                 valid_faces[i].person_id = person_map[label]
            
# # # # # # #             db.commit()
# # # # # # #             face_engine.rebuild_index(embeddings, [f.id for f in valid_faces])
# # # # # # #             logger.info(f"✅ Clustered {len(embeddings)} faces into {face_count} people")
# # # # # # #         else:
# # # # # # #             logger.warning("⚠️  No face embeddings found")

# # # # # # #         from clustering_engine import clustering_engine
# # # # # # #         all_images = db.query(DBImage).all()
# # # # # # #         album_count = 0
        
# # # # # # #         if all_images:
# # # # # # #             metadata = [
# # # # # # #                 {
# # # # # # #                     "id": img.id,
# # # # # # #                     "lat": img.lat or 0.0,
# # # # # # #                     "lon": img.lon or 0.0,
# # # # # # #                     "timestamp": img.timestamp
# # # # # # #                 }
# # # # # # #                 for img in all_images if img.timestamp
# # # # # # #             ]
            
# # # # # # #             if metadata:
# # # # # # #                 album_labels = clustering_engine.detect_events(metadata)
# # # # # # #                 album_map = {}
                
# # # # # # #                 for i, label in enumerate(album_labels):
# # # # # # #                     if label == -1:
# # # # # # #                         continue
                    
# # # # # # #                     if label not in album_map:
# # # # # # #                         cluster_imgs = [metadata[j] for j, l in enumerate(album_labels) if l == label]
# # # # # # #                         ts_list = [m['timestamp'] for m in cluster_imgs if m['timestamp']]
                        
# # # # # # #                         start_d = min(ts_list) if ts_list else None
# # # # # # #                         end_d = max(ts_list) if ts_list else None
                        
# # # # # # #                         new_album = Album(
# # # # # # #                             title=f"Event {label + 1}",
# # # # # # #                             type="event",
# # # # # # #                             start_date=start_d,
# # # # # # #                             end_date=end_d
# # # # # # #                         )
# # # # # # #                         db.add(new_album)
# # # # # # #                         db.flush()
# # # # # # #                         album_map[label] = new_album.id
# # # # # # #                         album_count += 1
                    
# # # # # # #                     all_images[i].album_id = album_map[label]
                
# # # # # # #                 db.commit()
# # # # # # #                 logger.info(f"✅ Created {album_count} albums")

# # # # # # #         return {
# # # # # # #             "status": "done",
# # # # # # #             "people": face_count,
# # # # # # #             "albums": album_count
# # # # # # #         }
    
# # # # # # #     except Exception as e:
# # # # # # #         db.rollback()
# # # # # # #         logger.error(f"❌ Recluster failed: {e}")
# # # # # # #         raise HTTPException(status_code=500, detail=str(e))
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # @app.post("/upload")
# # # # # # # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# # # # # # #     """Upload and process a single image"""
# # # # # # #     ext = os.path.splitext(file.filename)[1].lower()
# # # # # # #     if ext not in [".jpg", ".jpeg", ".png"]:
# # # # # # #         raise HTTPException(status_code=400, detail="Unsupported file format. Use JPG or PNG.")
    
# # # # # # #     filename = f"{uuid.uuid4()}{ext}"
# # # # # # #     file_path = os.path.join(IMAGE_DIR, filename)
    
# # # # # # #     db = SessionLocal()
    
# # # # # # #     try:
# # # # # # #         with open(file_path, "wb") as buffer:
# # # # # # #             shutil.copyfileobj(file.file, buffer)
        
# # # # # # #         logger.info(f"📤 Uploaded {filename}")
        
# # # # # # #         from PIL import Image as PILImage
# # # # # # #         try:
# # # # # # #             img_pil = PILImage.open(file_path)
# # # # # # #             width, height = img_pil.size
# # # # # # #         except:
# # # # # # #             width, height = None, None
        
# # # # # # #         try:
# # # # # # #             im_color = img_pil.convert('RGB') if 'img_pil' in locals() else PILImage.open(file_path).convert('RGB')
# # # # # # #             arr = np.array(im_color, dtype=np.float32)
# # # # # # #             avg = arr.mean(axis=(0, 1))
# # # # # # #             avg_r, avg_g, avg_b = float(avg[0]), float(avg[1]), float(avg[2])
# # # # # # #         except Exception:
# # # # # # #             avg_r = avg_g = avg_b = 0.0
        
# # # # # # #         clip_emb = None
# # # # # # #         try:
# # # # # # #             clip_emb = search_engine.get_image_embedding(file_path)
# # # # # # #             logger.info(f"✅ CLIP embedding extracted for {filename}")
# # # # # # #         except Exception as e:
# # # # # # #             logger.error(f"❌ CLIP embedding failed: {e}")

# # # # # # #         ocr_text = ""
# # # # # # #         try:
# # # # # # #             ocr_text = extract_text(file_path)
# # # # # # #             logger.info(f"✅ OCR completed: {len(ocr_text)} chars")
# # # # # # #         except Exception as e:
# # # # # # #             logger.error(f"❌ OCR failed: {e}")
        
# # # # # # #         person_count = 0
# # # # # # #         try:
# # # # # # #             person_count = detector_engine.detect_persons(file_path)
# # # # # # #             logger.info(f"✅ Detected {person_count} people")
# # # # # # #         except Exception as e:
# # # # # # #             logger.error(f"❌ Person detection failed: {e}")
        
# # # # # # #         img_record = DBImage(
# # # # # # #             filename=filename,
# # # # # # #             original_path=file_path,
# # # # # # #             timestamp=datetime.now(),
# # # # # # #             ocr_text=ocr_text,
# # # # # # #             person_count=person_count,
# # # # # # #             width=width,
# # # # # # #             avg_r=avg_r,
# # # # # # #             avg_g=avg_g,
# # # # # # #             avg_b=avg_b,
# # # # # # #             height=height,
# # # # # # #             size_bytes=os.path.getsize(file_path)
# # # # # # #         )
# # # # # # #         db.add(img_record)
# # # # # # #         db.flush()
        
# # # # # # #         logger.info(f"✅ Created image record ID={img_record.id}")
        
# # # # # # #         face_count = 0
# # # # # # #         try:
# # # # # # #             faces = face_engine.detect_faces(file_path)
# # # # # # #             logger.info(f"✅ Detected {len(faces)} faces")
            
# # # # # # #             for face in faces:
# # # # # # #                 emb = face['embedding'].astype(np.float32)
# # # # # # #                 emb_blob = emb.tobytes()

# # # # # # #                 assigned_person_id = None
# # # # # # #                 try:
# # # # # # #                     if face_engine.face_index is not None and face_engine.face_index.ntotal > 0:
# # # # # # #                         vec = emb.reshape(1, -1).astype('float32')
# # # # # # #                         faiss.normalize_L2(vec)
# # # # # # #                         D, I = face_engine.face_index.search(vec, FACE_MATCH_NEIGHBORS)
# # # # # # #                         votes = {}
# # # # # # #                         total_sim = 0.0
# # # # # # #                         for sim_val, idx_pos in zip(D[0], I[0]):
# # # # # # #                             if idx_pos == -1:
# # # # # # #                                 continue
# # # # # # #                             sim = float(sim_val)
# # # # # # #                             total_sim += sim
# # # # # # #                             try:
# # # # # # #                                 matched_face_db_id = face_engine.face_id_map[idx_pos]
# # # # # # #                                 matched_face = db.query(DBFace).filter(DBFace.id == int(matched_face_db_id)).first()
# # # # # # #                                 if matched_face and matched_face.person_id:
# # # # # # #                                     pid = matched_face.person_id
# # # # # # #                                     votes[pid] = votes.get(pid, 0.0) + sim
# # # # # # #                             except Exception:
# # # # # # #                                 continue

# # # # # # #                         if votes and total_sim > 0:
# # # # # # #                             best_pid, best_score = max(votes.items(), key=lambda x: x[1])
# # # # # # #                             if best_score / total_sim >= FACE_MATCH_VOTE_RATIO and best_score >= FACE_MATCH_THRESHOLD:
# # # # # # #                                 assigned_person_id = best_pid
# # # # # # #                 except Exception as e:
# # # # # # #                     logger.warning(f"Face matching check failed: {e}")

# # # # # # #                 if assigned_person_id is None:
# # # # # # #                     try:
# # # # # # #                         new_person = Person(name=f"Person")
# # # # # # #                         db.add(new_person)
# # # # # # #                         db.flush()
# # # # # # #                         assigned_person_id = new_person.id
# # # # # # #                     except Exception as e:
# # # # # # #                         logger.error(f"Could not create person record: {e}")
# # # # # # #                         assigned_person_id = None

# # # # # # #                 face_record = DBFace(
# # # # # # #                     image_id=img_record.id,
# # # # # # #                     bbox=str(face['bbox']),
# # # # # # #                     face_embedding=emb_blob,
# # # # # # #                     person_id=assigned_person_id
# # # # # # #                 )
# # # # # # #                 db.add(face_record)
# # # # # # #                 db.flush()

# # # # # # #                 try:
# # # # # # #                     face_engine.add_to_index(emb, face_record.id)
# # # # # # #                 except Exception as e:
# # # # # # #                     logger.error(f"Failed to add face to index: {e}")

# # # # # # #                 face_count += 1
# # # # # # #                 logger.info(f"✅ Processed and indexed face {face_count} (person_id={assigned_person_id})")
        
# # # # # # #         except Exception as e:
# # # # # # #             logger.error(f"❌ Face detection failed: {e}")
        
# # # # # # #         if clip_emb is not None:
# # # # # # #             try:
# # # # # # #                 if search_engine.index is None:
# # # # # # #                     dim = clip_emb.shape[0]
# # # # # # #                     sub_index = faiss.IndexFlatIP(dim)
# # # # # # #                     search_engine.index = faiss.IndexIDMap(sub_index)
                
# # # # # # #                 new_vec = clip_emb.reshape(1, -1).astype('float32')
# # # # # # #                 faiss.normalize_L2(new_vec)
                
# # # # # # #                 ids_np = np.array([img_record.id]).astype('int64')
# # # # # # #                 search_engine.index.add_with_ids(new_vec, ids_np)
                
# # # # # # #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# # # # # # #                 logger.info(f"✅ Updated image FAISS index")
# # # # # # #             except Exception as e:
# # # # # # #                 logger.error(f"❌ Index update failed: {e}")
        
# # # # # # #         db.commit()

# # # # # # #         if background_tasks is not None:
# # # # # # #             should_trigger_recluster(background_tasks)

# # # # # # #         return {
# # # # # # #             "status": "success",
# # # # # # #             "id": img_record.id,
# # # # # # #             "filename": filename,
# # # # # # #             "person_count": person_count,
# # # # # # #             "face_count": face_count
# # # # # # #         }
    
# # # # # # #     except Exception as e:
# # # # # # #         db.rollback()
# # # # # # #         if os.path.exists(file_path):
# # # # # # #             os.remove(file_path)
# # # # # # #         logger.error(f"❌ Upload failed: {e}")
# # # # # # #         raise HTTPException(status_code=500, detail=str(e))
    
# # # # # # #     finally:
# # # # # # #         db.close()

# # # # # # # if __name__ == "__main__":
# # # # # # #     import uvicorn
# # # # # # #     uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# # # # # # # FINAL main.py - Smart Two-Pass Search
# # # # # # # Pass 1: Find keyword-confirmed images, get minimum score
# # # # # # # Pass 2: Use that minimum as threshold for ALL images
# # # # # # # Result: NO unrelated images after good ones!

# # # # # # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# # # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # # from fastapi.staticfiles import StaticFiles
# # # # # # import os, uuid, shutil, numpy as np, faiss
# # # # # # from datetime import datetime
# # # # # # import logging
# # # # # # from contextlib import asynccontextmanager
# # # # # # import datetime as datetime_module

# # # # # # logging.basicConfig(level=logging.INFO)
# # # # # # logger = logging.getLogger("main")

# # # # # # from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
# # # # # # from search_engine import search_engine, resolve_query
# # # # # # from voice_engine import voice_engine
# # # # # # from face_engine import face_engine
# # # # # # from ocr_engine import extract_text
# # # # # # from detector_engine import detector_engine
# # # # # # from duplicate_engine import duplicate_engine

# # # # # # IMAGE_DIR = "../data/images"
# # # # # # FAISS_INDEX_PATH = "../data/index.faiss"

# # # # # # FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
# # # # # # FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
# # # # # # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # # # # # RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # # # # # RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # # # # # CLIP_SCORE_MIN = 0.10

# # # # # # RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # # # # # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # # # # # recluster_last_triggered = None

# # # # # # def should_trigger_recluster(background_tasks):
# # # # # #     global recluster_last_triggered
# # # # # #     if not RECLUSTER_ON_UPLOAD or not background_tasks:
# # # # # #         return
# # # # # #     try:
# # # # # #         counter = 0
# # # # # #         if os.path.exists(RECLUSTER_COUNTER_PATH):
# # # # # #             try:
# # # # # #                 with open(RECLUSTER_COUNTER_PATH, 'r') as f:
# # # # # #                     counter = int(f.read().strip())
# # # # # #             except:
# # # # # #                 pass
# # # # # #         counter += 1
# # # # # #         with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # # # #             f.write(str(counter))
# # # # # #         should_trigger = counter >= RECLUSTER_BATCH_SIZE
# # # # # #         now = datetime_module.datetime.now()
# # # # # #         if recluster_last_triggered:
# # # # # #             elapsed = (now - recluster_last_triggered).total_seconds()
# # # # # #             if elapsed >= RECLUSTER_TIMER_SECONDS:
# # # # # #                 should_trigger = True
# # # # # #         elif counter > 0:
# # # # # #             should_trigger = counter >= RECLUSTER_BATCH_SIZE
# # # # # #         if should_trigger:
# # # # # #             logger.info(f"📊 Recluster triggered")
# # # # # #             background_tasks.add_task(recluster)
# # # # # #             recluster_last_triggered = now
# # # # # #             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # # # #                 f.write('0')
# # # # # #     except Exception as e:
# # # # # #         logger.warning(f"Recluster check failed: {e}")

# # # # # # @asynccontextmanager
# # # # # # async def lifespan(app: FastAPI):
# # # # # #     init_db()
# # # # # #     logger.info("🚀 Starting with SMART TWO-PASS search...")
# # # # # #     if os.path.exists(FAISS_INDEX_PATH):
# # # # # #         try:
# # # # # #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# # # # # #             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
# # # # # #         except Exception as e:
# # # # # #             logger.error(f"Index load failed: {e}")
# # # # # #             search_engine.index = None
# # # # # #     logger.info("✅ Ready!")
# # # # # #     yield

# # # # # # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)
# # # # # # app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# # # # # # if not os.path.exists(IMAGE_DIR):
# # # # # #     os.makedirs(IMAGE_DIR)
# # # # # # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # # # # # @app.get("/health")
# # # # # # def health():
# # # # # #     return {"status": "ready", "mode": "SMART_TWO_PASS", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# # # # # # @app.get("/test-db")
# # # # # # def test_db():
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         count = db.query(DBImage).count()
# # # # # #         images = db.query(DBImage).limit(1).all()
# # # # # #         return {"status": "ok", "total_images": count, "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None}
# # # # # #     except Exception as e:
# # # # # #         return {"status": "error", "message": str(e)}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.post("/search")
# # # # # # def search(query: str = Form(...), top_k: int = Form(20)):
# # # # # #     """
# # # # # #     SMART TWO-PASS SEARCH:
# # # # # #     Pass 1: Find keyword-confirmed images, get minimum CLIP score
# # # # # #     Pass 2: Use that minimum as threshold for ALL images
# # # # # #     Result: ONLY relevant images shown!
# # # # # #     """
# # # # # #     if not query or not query.strip():
# # # # # #         return {"status": "error", "message": "Query empty"}

# # # # # #     processed_query = resolve_query(query)
# # # # # #     logger.info(f"🔍 Search: '{query}'")

# # # # # #     COLOR_MAP = {
# # # # # #         'red': (1.0, 0, 0), 'blue': (0, 0, 1.0), 'green': (0, 1.0, 0),
# # # # # #         'yellow': (1.0, 1.0, 0), 'orange': (1.0, 0.5, 0), 'purple': (0.5, 0, 0.5),
# # # # # #         'pink': (1.0, 0.75, 0.8), 'black': (0, 0, 0), 'white': (1, 1, 1),
# # # # # #         'gray': (0.5, 0.5, 0.5), 'brown': (0.6, 0.4, 0.2)
# # # # # #     }
# # # # # #     query_lower = query.lower()
# # # # # #     query_colors = [rgb for name, rgb in COLOR_MAP.items() if name in query_lower]

# # # # # #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# # # # # #     if query_emb is None or search_engine.index is None:
# # # # # #         return {"status": "error", "message": "No images indexed"}

# # # # # #     candidate_k = min(top_k * 8, 250)
# # # # # #     query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
# # # # # #     faiss.normalize_L2(query_emb_reshaped)
# # # # # #     distances, indices = search_engine.index.search(query_emb_reshaped, candidate_k)

# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         # ===== PASS 1: Collect ALL candidates with bonuses =====
# # # # # #         all_candidates = []

# # # # # #         for dist, idx in zip(distances[0], indices[0]):
# # # # # #             if idx == -1:
# # # # # #                 continue
# # # # # #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# # # # # #             if not img:
# # # # # #                 continue

# # # # # #             raw_sim = float(dist)
# # # # # #             clip_score = max(0.0, raw_sim)
            
# # # # # #             if clip_score < CLIP_SCORE_MIN:
# # # # # #                 continue

# # # # # #             # OCR matching
# # # # # #             ocr_text = (img.ocr_text or "").lower()
# # # # # #             query_words = processed_query.lower().split()
# # # # # #             significant_words = [w for w in query_words if len(w) > 2]
            
# # # # # #             ocr_bonus = 0.0
# # # # # #             if significant_words and ocr_text:
# # # # # #                 matches = sum(1 for w in significant_words if w in ocr_text)
# # # # # #                 if matches > 0:
# # # # # #                     ocr_bonus = min(matches / len(significant_words), 1.0)

# # # # # #             # Tag matching
# # # # # #             tag_bonus = 0.0
# # # # # #             if img.scene_label:
# # # # # #                 tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
# # # # # #                 query_objects = [w for w in query_words if len(w) > 2]
# # # # # #                 if query_objects:
# # # # # #                     for obj in query_objects:
# # # # # #                         if any(obj in tag for tag in tags):
# # # # # #                             tag_bonus = 1.0
# # # # # #                             break

# # # # # #             # Color matching
# # # # # #             color_bonus = 0.0
# # # # # #             if query_colors and getattr(img, 'avg_r', None) is not None:
# # # # # #                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# # # # # #                 for qc in query_colors:
# # # # # #                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# # # # # #                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
# # # # # #                     color_bonus = max(color_bonus, score)

# # # # # #             has_keyword_match = ocr_bonus > 0 or tag_bonus > 0
            
# # # # # #             if has_keyword_match:
# # # # # #                 final_score = (
# # # # # #                     (0.40 * clip_score) +
# # # # # #                     (0.35 * ocr_bonus) +
# # # # # #                     (0.15 * color_bonus) +
# # # # # #                     (0.10 * tag_bonus)
# # # # # #                 )
# # # # # #             else:
# # # # # #                 final_score = clip_score

# # # # # #             all_candidates.append({
# # # # # #                 'img': img,
# # # # # #                 'clip_score': clip_score,
# # # # # #                 'final_score': final_score,
# # # # # #                 'ocr_bonus': ocr_bonus,
# # # # # #                 'tag_bonus': tag_bonus,
# # # # # #                 'color_bonus': color_bonus,
# # # # # #                 'has_keyword_match': has_keyword_match
# # # # # #             })

# # # # # #         # ===== CALCULATE THRESHOLD =====
# # # # # #         keyword_confirmed_clips = [c['clip_score'] for c in all_candidates if c['has_keyword_match']]

# # # # # #         if keyword_confirmed_clips:
# # # # # #             # Use minimum keyword-confirmed as threshold
# # # # # #             threshold = min(keyword_confirmed_clips)
# # # # # #             logger.info(f"🎯 Threshold from {len(keyword_confirmed_clips)} confirmed: {threshold:.4f}")
# # # # # #         else:
# # # # # #             # No keywords - use adaptive fallback
# # # # # #             all_clips = [c['clip_score'] for c in all_candidates]
# # # # # #             if all_clips:
# # # # # #                 best_clip = max(all_clips)
# # # # # #                 threshold = best_clip - 0.05
# # # # # #                 logger.info(f"⚠️ Adaptive threshold: {threshold:.4f}")
# # # # # #             else:
# # # # # #                 return {"status": "not_found", "message": f"No images for '{query}'"}

# # # # # #         # ===== PASS 2: Filter by threshold =====
# # # # # #         results = []

# # # # # #         for candidate in all_candidates:
# # # # # #             img = candidate['img']
# # # # # #             clip_score = candidate['clip_score']
# # # # # #             final_score = candidate['final_score']
# # # # # #             ocr_bonus = candidate['ocr_bonus']
# # # # # #             tag_bonus = candidate['tag_bonus']
# # # # # #             has_keyword_match = candidate['has_keyword_match']
            
# # # # # #             # Check threshold
# # # # # #             if has_keyword_match:
# # # # # #                 show = True
# # # # # #             else:
# # # # # #                 show = clip_score >= threshold
            
# # # # # #             if not show:
# # # # # #                 logger.debug(f"❌ Skip: CLIP {clip_score:.4f} < {threshold:.4f}")
# # # # # #                 continue
            
# # # # # #             logger.debug(f"✅ {img.filename}: CLIP={clip_score:.4f}, FINAL={final_score:.4f}")
            
# # # # # #             results.append({
# # # # # #                 "id": img.id,
# # # # # #                 "filename": img.filename,
# # # # # #                 "score": round(final_score * 100, 2),
# # # # # #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # # # # #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # # # # #                 "person_count": img.person_count or 0
# # # # # #             })

# # # # # #         results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
# # # # # #         if not results:
# # # # # #             return {"status": "not_found", "message": f"No match for '{query}'"}

# # # # # #         logger.info(f"✅ Found {len(results)} results")
# # # # # #         return {"status": "found", "query": query, "count": len(results), "threshold": round(threshold * 100, 1), "results": results}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.post("/search/voice")
# # # # # # def voice_search(duration: int = Form(5)):
# # # # # #     try:
# # # # # #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# # # # # #         if not transcribed:
# # # # # #             return {"status": "error", "message": "No audio"}
# # # # # #         return search(query=transcribed, top_k=20)
# # # # # #     except Exception as e:
# # # # # #         return {"status": "error", "message": str(e)}

# # # # # # @app.get("/timeline")
# # # # # # def get_timeline():
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
# # # # # #         results = [{"id": img.id, "filename": img.filename, "date": img.timestamp.isoformat() if img.timestamp else None, "thumbnail": f"/images/{img.filename}"} for img in images]
# # # # # #         return {"count": len(results), "results": results}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.get("/faces")
# # # # # # def get_faces(person_id: int = Query(None)):
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         if person_id:
# # # # # #             person = db.query(Person).filter(Person.id == person_id).first()
# # # # # #             if not person:
# # # # # #                 raise HTTPException(status_code=404)
# # # # # #             faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# # # # # #             images = [{"id": f.image_id, "filename": db.query(DBImage).filter(DBImage.id == f.image_id).first().filename} for f in faces if db.query(DBImage).filter(DBImage.id == f.image_id).first()]
# # # # # #             return {"id": person.id, "name": person.name, "face_count": len(faces), "images": images}
# # # # # #         else:
# # # # # #             people = db.query(Person).all()
# # # # # #             results = []
# # # # # #             for p in people:
# # # # # #                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# # # # # #                 if faces:
# # # # # #                     results.append({"id": p.id, "name": p.name, "count": len(faces)})
# # # # # #             return {"results": results, "count": len(results)}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.post("/people/{person_id}")
# # # # # # def update_person(person_id: int, name: str = Form(...)):
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # # # #         if not person:
# # # # # #             raise HTTPException(status_code=404)
# # # # # #         person.name = name
# # # # # #         db.commit()
# # # # # #         return {"status": "success", "id": person.id, "name": person.name}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.get("/people/{person_id}/celebcheck")
# # # # # # def check_celebrity_match(person_id: int):
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # # # #         if not person:
# # # # # #             return {"status": "no_match"}
# # # # # #         return {"status": "no_match"}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.get("/albums")
# # # # # # def get_albums(album_id: int = Query(None)):
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         if album_id:
# # # # # #             album = db.query(Album).filter(Album.id == album_id).first()
# # # # # #             if not album:
# # # # # #                 raise HTTPException(status_code=404)
# # # # # #             images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
# # # # # #             return {"id": album.id, "title": album.title, "count": len(images)}
# # # # # #         else:
# # # # # #             albums = db.query(Album).all()
# # # # # #             results = [{"id": a.id, "title": a.title, "count": db.query(DBImage).filter(DBImage.album_id == a.id).count()} for a in albums]
# # # # # #             return {"results": results, "count": len(results)}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.post("/favorites")
# # # # # # def add_favorite(image_id: int = Form(...)):
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # # # # #         if not img:
# # # # # #             raise HTTPException(status_code=404)
# # # # # #         img.is_favorite = not getattr(img, 'is_favorite', False)
# # # # # #         db.commit()
# # # # # #         return {"status": "success"}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.get("/favorites")
# # # # # # def get_favorites():
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         images = db.query(DBImage).filter(DBImage.is_favorite == True).all()
# # # # # #         return {"count": len(images), "results": [{"id": img.id, "filename": img.filename} for img in images]}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.get("/duplicates")
# # # # # # def get_duplicates():
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         all_images = db.query(DBImage).all()
# # # # # #         return {"status": "found", "duplicate_groups": [], "total_groups": 0}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.post("/recluster")
# # # # # # def recluster():
# # # # # #     db = SessionLocal()
# # # # # #     try:
# # # # # #         logger.info("🔄 Recluster...")
# # # # # #         db.query(DBFace).update({"person_id": None})
# # # # # #         db.query(Person).delete()
# # # # # #         db.query(DBImage).update({"album_id": None})
# # # # # #         db.query(Album).filter(Album.type == "event").delete()
# # # # # #         db.commit()
# # # # # #         return {"status": "done", "people": 0, "albums": 0}
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # @app.post("/upload")
# # # # # # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# # # # # #     ext = os.path.splitext(file.filename)[1].lower()
# # # # # #     if ext not in [".jpg", ".jpeg", ".png"]:
# # # # # #         raise HTTPException(status_code=400)
    
# # # # # #     filename = f"{uuid.uuid4()}{ext}"
# # # # # #     file_path = os.path.join(IMAGE_DIR, filename)
# # # # # #     db = SessionLocal()
    
# # # # # #     try:
# # # # # #         with open(file_path, "wb") as buffer:
# # # # # #             shutil.copyfileobj(file.file, buffer)
        
# # # # # #         from PIL import Image as PILImage
# # # # # #         try:
# # # # # #             img_pil = PILImage.open(file_path)
# # # # # #             width, height = img_pil.size
# # # # # #         except:
# # # # # #             width, height = None, None
        
# # # # # #         avg_r = avg_g = avg_b = 0.0
        
# # # # # #         clip_emb = None
# # # # # #         try:
# # # # # #             clip_emb = search_engine.get_image_embedding(file_path)
# # # # # #         except:
# # # # # #             pass

# # # # # #         ocr_text = ""
# # # # # #         try:
# # # # # #             ocr_text = extract_text(file_path)
# # # # # #         except:
# # # # # #             pass
        
# # # # # #         person_count = 0
# # # # # #         try:
# # # # # #             person_count = detector_engine.detect_persons(file_path)
# # # # # #         except:
# # # # # #             pass
        
# # # # # #         img_record = DBImage(
# # # # # #             filename=filename, original_path=file_path, timestamp=datetime.now(),
# # # # # #             ocr_text=ocr_text, person_count=person_count, width=width,
# # # # # #             avg_r=avg_r, avg_g=avg_g, avg_b=avg_b, height=height
# # # # # #         )
# # # # # #         db.add(img_record)
# # # # # #         db.flush()
        
# # # # # #         if clip_emb is not None:
# # # # # #             try:
# # # # # #                 if search_engine.index is None:
# # # # # #                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
# # # # # #                 new_vec = clip_emb.reshape(1, -1).astype('float32')
# # # # # #                 faiss.normalize_L2(new_vec)
# # # # # #                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id]).astype('int64'))
# # # # # #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# # # # # #             except:
# # # # # #                 pass
        
# # # # # #         db.commit()

# # # # # #         return {"status": "success", "id": img_record.id, "filename": filename}
# # # # # #     except Exception as e:
# # # # # #         db.rollback()
# # # # # #         if os.path.exists(file_path):
# # # # # #             os.remove(file_path)
# # # # # #         raise HTTPException(status_code=500)
# # # # # #     finally:
# # # # # #         db.close()

# # # # # # if __name__ == "__main__":
# # # # # #     import uvicorn
# # # # # #     uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


# # # # # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # from fastapi.staticfiles import StaticFiles
# # # # # import os, uuid, shutil, numpy as np, faiss
# # # # # from datetime import datetime
# # # # # import logging
# # # # # from contextlib import asynccontextmanager
# # # # # import datetime as datetime_module

# # # # # logging.basicConfig(level=logging.INFO)
# # # # # logger = logging.getLogger("main")

# # # # # from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
# # # # # from search_engine import search_engine, resolve_query
# # # # # from voice_engine import voice_engine
# # # # # from face_engine import face_engine
# # # # # from ocr_engine import extract_text
# # # # # from detector_engine import detector_engine
# # # # # from duplicate_engine import duplicate_engine

# # # # # IMAGE_DIR = "../data/images"
# # # # # FAISS_INDEX_PATH = "../data/index.faiss"

# # # # # FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
# # # # # FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
# # # # # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # # # # RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # # # # RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # # # # # ── Search quality thresholds ─────────────────────────────────────────────────
# # # # # # Hard floor: no image below this CLIP score ever appears in results.
# # # # # # CLIP ViT-B/32 cosine similarity guide:
# # # # # #   ≥ 0.28  strong semantic match
# # # # # #   0.22–0.28  plausible match
# # # # # #   < 0.22  usually unrelated
# # # # # CLIP_SCORE_MIN    = float(os.environ.get("CLIP_SCORE_MIN",    0.22))

# # # # # # When keyword-confirmed images anchor the threshold, never let it fall below
# # # # # # this value (prevents one low-quality tagged image from flooding results).
# # # # # THRESHOLD_FLOOR   = float(os.environ.get("THRESHOLD_FLOOR",   0.22))

# # # # # # Adaptive mode (no keyword hits): only show images within this fraction of
# # # # # # the top CLIP score. 0.92 means "within 8% of the best match".
# # # # # ADAPTIVE_RATIO    = float(os.environ.get("ADAPTIVE_RATIO",    0.92))

# # # # # RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # # # # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # # # # recluster_last_triggered = None

# # # # # def should_trigger_recluster(background_tasks):
# # # # #     global recluster_last_triggered
# # # # #     if not RECLUSTER_ON_UPLOAD or not background_tasks:
# # # # #         return
# # # # #     try:
# # # # #         counter = 0
# # # # #         if os.path.exists(RECLUSTER_COUNTER_PATH):
# # # # #             try:
# # # # #                 with open(RECLUSTER_COUNTER_PATH, 'r') as f:
# # # # #                     counter = int(f.read().strip())
# # # # #             except:
# # # # #                 pass
# # # # #         counter += 1
# # # # #         with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # # #             f.write(str(counter))
# # # # #         should_trigger = counter >= RECLUSTER_BATCH_SIZE
# # # # #         now = datetime_module.datetime.now()
# # # # #         if recluster_last_triggered:
# # # # #             elapsed = (now - recluster_last_triggered).total_seconds()
# # # # #             if elapsed >= RECLUSTER_TIMER_SECONDS:
# # # # #                 should_trigger = True
# # # # #         elif counter > 0:
# # # # #             should_trigger = counter >= RECLUSTER_BATCH_SIZE
# # # # #         if should_trigger:
# # # # #             logger.info(f"📊 Recluster triggered")
# # # # #             background_tasks.add_task(recluster)
# # # # #             recluster_last_triggered = now
# # # # #             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # # #                 f.write('0')
# # # # #     except Exception as e:
# # # # #         logger.warning(f"Recluster check failed: {e}")

# # # # # @asynccontextmanager
# # # # # async def lifespan(app: FastAPI):
# # # # #     init_db()
# # # # #     logger.info("🚀 Starting with SMART TWO-PASS search...")
# # # # #     if os.path.exists(FAISS_INDEX_PATH):
# # # # #         try:
# # # # #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# # # # #             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
# # # # #         except Exception as e:
# # # # #             logger.error(f"Index load failed: {e}")
# # # # #             search_engine.index = None
# # # # #     logger.info("✅ Ready!")
# # # # #     yield

# # # # # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)
# # # # # app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# # # # # if not os.path.exists(IMAGE_DIR):
# # # # #     os.makedirs(IMAGE_DIR)
# # # # # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # # # # @app.get("/health")
# # # # # def health():
# # # # #     return {"status": "ready", "mode": "SMART_TWO_PASS", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# # # # # @app.get("/test-db")
# # # # # def test_db():
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         count = db.query(DBImage).count()
# # # # #         images = db.query(DBImage).limit(1).all()
# # # # #         return {"status": "ok", "total_images": count, "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None}
# # # # #     except Exception as e:
# # # # #         return {"status": "error", "message": str(e)}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.post("/search")
# # # # # def search(query: str = Form(...), top_k: int = Form(20)):
# # # # #     """
# # # # #     SMART TWO-PASS SEARCH:
# # # # #     Pass 1: Find keyword-confirmed images, get minimum CLIP score
# # # # #     Pass 2: Use that minimum as threshold for ALL images
# # # # #     Result: ONLY relevant images shown!
# # # # #     """
# # # # #     if not query or not query.strip():
# # # # #         return {"status": "error", "message": "Query empty"}

# # # # #     processed_query = resolve_query(query)
# # # # #     logger.info(f"🔍 Search: '{query}'")

# # # # #     COLOR_MAP = {
# # # # #         'red': (1.0, 0, 0), 'blue': (0, 0, 1.0), 'green': (0, 1.0, 0),
# # # # #         'yellow': (1.0, 1.0, 0), 'orange': (1.0, 0.5, 0), 'purple': (0.5, 0, 0.5),
# # # # #         'pink': (1.0, 0.75, 0.8), 'black': (0, 0, 0), 'white': (1, 1, 1),
# # # # #         'gray': (0.5, 0.5, 0.5), 'brown': (0.6, 0.4, 0.2)
# # # # #     }
# # # # #     query_lower = query.lower()
# # # # #     query_colors = [rgb for name, rgb in COLOR_MAP.items() if name in query_lower]

# # # # #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# # # # #     if query_emb is None or search_engine.index is None:
# # # # #         return {"status": "error", "message": "No images indexed"}

# # # # #     candidate_k = min(top_k * 8, 250)
# # # # #     query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
# # # # #     faiss.normalize_L2(query_emb_reshaped)
# # # # #     distances, indices = search_engine.index.search(query_emb_reshaped, candidate_k)

# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         # ===== PASS 1: Collect ALL candidates with bonuses =====
# # # # #         all_candidates = []

# # # # #         for dist, idx in zip(distances[0], indices[0]):
# # # # #             if idx == -1:
# # # # #                 continue
# # # # #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# # # # #             if not img:
# # # # #                 continue

# # # # #             raw_sim = float(dist)
# # # # #             clip_score = max(0.0, raw_sim)
            
# # # # #             if clip_score < CLIP_SCORE_MIN:
# # # # #                 continue

# # # # #             # OCR matching
# # # # #             ocr_text = (img.ocr_text or "").lower()
# # # # #             query_words = processed_query.lower().split()
# # # # #             significant_words = [w for w in query_words if len(w) > 2]
            
# # # # #             ocr_bonus = 0.0
# # # # #             if significant_words and ocr_text:
# # # # #                 matches = sum(1 for w in significant_words if w in ocr_text)
# # # # #                 if matches > 0:
# # # # #                     ocr_bonus = min(matches / len(significant_words), 1.0)

# # # # #             # Tag matching — EXACT word match only.
# # # # #             # Old code used `obj in tag` (substring), which caused:
# # # # #             #   "dog" matching "hotdog", "man" matching "woman" / "human", etc.
# # # # #             # Fix: split each tag into words and check for exact membership.
# # # # #             tag_bonus = 0.0
# # # # #             if img.scene_label:
# # # # #                 tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
# # # # #                 # Flatten tags into individual words for exact lookup
# # # # #                 tag_words = set()
# # # # #                 for tag in tags:
# # # # #                     for word in tag.split():
# # # # #                         tag_words.add(word)
# # # # #                 query_objects = [w for w in query_words if len(w) > 2]
# # # # #                 if query_objects:
# # # # #                     for obj in query_objects:
# # # # #                         if obj in tag_words:   # exact word match only
# # # # #                             tag_bonus = 1.0
# # # # #                             break

# # # # #             # Color matching
# # # # #             color_bonus = 0.0
# # # # #             if query_colors and getattr(img, 'avg_r', None) is not None:
# # # # #                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# # # # #                 for qc in query_colors:
# # # # #                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# # # # #                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
# # # # #                     color_bonus = max(color_bonus, score)

# # # # #             has_keyword_match = ocr_bonus > 0 or tag_bonus > 0
            
# # # # #             if has_keyword_match:
# # # # #                 final_score = (
# # # # #                     (0.40 * clip_score) +
# # # # #                     (0.35 * ocr_bonus) +
# # # # #                     (0.15 * color_bonus) +
# # # # #                     (0.10 * tag_bonus)
# # # # #                 )
# # # # #             else:
# # # # #                 final_score = clip_score

# # # # #             all_candidates.append({
# # # # #                 'img': img,
# # # # #                 'clip_score': clip_score,
# # # # #                 'final_score': final_score,
# # # # #                 'ocr_bonus': ocr_bonus,
# # # # #                 'tag_bonus': tag_bonus,
# # # # #                 'color_bonus': color_bonus,
# # # # #                 'has_keyword_match': has_keyword_match
# # # # #             })

# # # # #         # ===== CALCULATE THRESHOLD =====
# # # # #         keyword_confirmed_clips = [c['clip_score'] for c in all_candidates if c['has_keyword_match']]

# # # # #         if keyword_confirmed_clips:
# # # # #             # Anchor threshold on confirmed images — but never below THRESHOLD_FLOOR.
# # # # #             # Old code: threshold = min(keyword_confirmed_clips)  → could be as low as 0.10
# # # # #             # Fix: use the minimum confirmed score, but enforce the hard floor.
# # # # #             threshold = max(min(keyword_confirmed_clips), THRESHOLD_FLOOR)
# # # # #             logger.info(f"🎯 Threshold from {len(keyword_confirmed_clips)} confirmed: {threshold:.4f}")
# # # # #         else:
# # # # #             # No keyword hits — stay close to the best CLIP score so only
# # # # #             # semantically similar images appear.
# # # # #             # Old code: threshold = best_clip - 0.05  (too wide a window)
# # # # #             # Fix: threshold = best_clip * ADAPTIVE_RATIO, floored at THRESHOLD_FLOOR
# # # # #             all_clips = [c['clip_score'] for c in all_candidates]
# # # # #             if all_clips:
# # # # #                 best_clip = max(all_clips)
# # # # #                 threshold = max(best_clip * ADAPTIVE_RATIO, THRESHOLD_FLOOR)
# # # # #                 logger.info(f"⚠️  Adaptive threshold: {threshold:.4f} (best={best_clip:.4f})")
# # # # #             else:
# # # # #                 return {"status": "not_found", "message": f"No images for '{query}'"}

# # # # #         # ===== PASS 2: Filter by threshold =====
# # # # #         results = []

# # # # #         for candidate in all_candidates:
# # # # #             img = candidate['img']
# # # # #             clip_score = candidate['clip_score']
# # # # #             final_score = candidate['final_score']
# # # # #             has_keyword_match = candidate['has_keyword_match']

# # # # #             # Every image — keyword-confirmed or not — must reach the threshold.
# # # # #             # Old code gave keyword-matched images a free pass (show = True regardless
# # # # #             # of clip_score), letting low-quality tag matches flood the results.
# # # # #             # Fix: threshold applies uniformly; keyword match only affects score weight.
# # # # #             if clip_score < threshold:
# # # # #                 logger.debug(f"❌ Skip {img.filename}: CLIP {clip_score:.4f} < {threshold:.4f}")
# # # # #                 continue
            
# # # # #             logger.debug(f"✅ {img.filename}: CLIP={clip_score:.4f}, FINAL={final_score:.4f}")
            
# # # # #             results.append({
# # # # #                 "id": img.id,
# # # # #                 "filename": img.filename,
# # # # #                 "score": round(final_score * 100, 2),
# # # # #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # # # #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # # # #                 "person_count": img.person_count or 0
# # # # #             })

# # # # #         results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
# # # # #         if not results:
# # # # #             return {"status": "not_found", "message": f"No match for '{query}'"}

# # # # #         logger.info(f"✅ Found {len(results)} results")
# # # # #         return {"status": "found", "query": query, "count": len(results), "threshold": round(threshold * 100, 1), "results": results}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.post("/search/voice")
# # # # # def voice_search(duration: int = Form(5)):
# # # # #     try:
# # # # #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# # # # #         if not transcribed:
# # # # #             return {"status": "error", "message": "No audio"}
# # # # #         return search(query=transcribed, top_k=20)
# # # # #     except Exception as e:
# # # # #         return {"status": "error", "message": str(e)}

# # # # # @app.get("/timeline")
# # # # # def get_timeline():
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
# # # # #         results = [{"id": img.id, "filename": img.filename, "date": img.timestamp.isoformat() if img.timestamp else None, "thumbnail": f"/images/{img.filename}"} for img in images]
# # # # #         return {"count": len(results), "results": results}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.get("/faces")
# # # # # def get_faces(person_id: int = Query(None)):
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         if person_id:
# # # # #             person = db.query(Person).filter(Person.id == person_id).first()
# # # # #             if not person:
# # # # #                 raise HTTPException(status_code=404)
# # # # #             faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# # # # #             images = [{"id": f.image_id, "filename": db.query(DBImage).filter(DBImage.id == f.image_id).first().filename} for f in faces if db.query(DBImage).filter(DBImage.id == f.image_id).first()]
# # # # #             return {"id": person.id, "name": person.name, "face_count": len(faces), "images": images}
# # # # #         else:
# # # # #             people = db.query(Person).all()
# # # # #             results = []
# # # # #             for p in people:
# # # # #                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# # # # #                 if faces:
# # # # #                     results.append({"id": p.id, "name": p.name, "count": len(faces)})
# # # # #             return {"results": results, "count": len(results)}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.post("/people/{person_id}")
# # # # # def update_person(person_id: int, name: str = Form(...)):
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # # #         if not person:
# # # # #             raise HTTPException(status_code=404)
# # # # #         person.name = name
# # # # #         db.commit()
# # # # #         return {"status": "success", "id": person.id, "name": person.name}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.get("/people/{person_id}/celebcheck")
# # # # # def check_celebrity_match(person_id: int):
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # # #         if not person:
# # # # #             return {"status": "no_match"}
# # # # #         return {"status": "no_match"}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.get("/albums")
# # # # # def get_albums(album_id: int = Query(None)):
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         if album_id:
# # # # #             album = db.query(Album).filter(Album.id == album_id).first()
# # # # #             if not album:
# # # # #                 raise HTTPException(status_code=404)
# # # # #             images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
# # # # #             return {"id": album.id, "title": album.title, "count": len(images)}
# # # # #         else:
# # # # #             albums = db.query(Album).all()
# # # # #             results = [{"id": a.id, "title": a.title, "count": db.query(DBImage).filter(DBImage.album_id == a.id).count()} for a in albums]
# # # # #             return {"results": results, "count": len(results)}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.post("/favorites")
# # # # # def add_favorite(image_id: int = Form(...)):
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # # # #         if not img:
# # # # #             raise HTTPException(status_code=404)
# # # # #         img.is_favorite = not getattr(img, 'is_favorite', False)
# # # # #         db.commit()
# # # # #         return {"status": "success"}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.get("/favorites")
# # # # # def get_favorites():
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         images = db.query(DBImage).filter(DBImage.is_favorite == True).all()
# # # # #         return {"count": len(images), "results": [{"id": img.id, "filename": img.filename} for img in images]}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.get("/duplicates")
# # # # # def get_duplicates():
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         all_images = db.query(DBImage).all()
# # # # #         return {"status": "found", "duplicate_groups": [], "total_groups": 0}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.post("/recluster")
# # # # # def recluster():
# # # # #     db = SessionLocal()
# # # # #     try:
# # # # #         logger.info("🔄 Recluster...")
# # # # #         db.query(DBFace).update({"person_id": None})
# # # # #         db.query(Person).delete()
# # # # #         db.query(DBImage).update({"album_id": None})
# # # # #         db.query(Album).filter(Album.type == "event").delete()
# # # # #         db.commit()
# # # # #         return {"status": "done", "people": 0, "albums": 0}
# # # # #     finally:
# # # # #         db.close()

# # # # # @app.post("/upload")
# # # # # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# # # # #     ext = os.path.splitext(file.filename)[1].lower()
# # # # #     if ext not in [".jpg", ".jpeg", ".png"]:
# # # # #         raise HTTPException(status_code=400)
    
# # # # #     filename = f"{uuid.uuid4()}{ext}"
# # # # #     file_path = os.path.join(IMAGE_DIR, filename)
# # # # #     db = SessionLocal()
    
# # # # #     try:
# # # # #         with open(file_path, "wb") as buffer:
# # # # #             shutil.copyfileobj(file.file, buffer)
        
# # # # #         from PIL import Image as PILImage
# # # # #         try:
# # # # #             img_pil = PILImage.open(file_path)
# # # # #             width, height = img_pil.size
# # # # #         except:
# # # # #             width, height = None, None
        
# # # # #         avg_r = avg_g = avg_b = 0.0
        
# # # # #         clip_emb = None
# # # # #         try:
# # # # #             clip_emb = search_engine.get_image_embedding(file_path)
# # # # #         except:
# # # # #             pass

# # # # #         ocr_text = ""
# # # # #         try:
# # # # #             ocr_text = extract_text(file_path)
# # # # #         except:
# # # # #             pass
        
# # # # #         person_count = 0
# # # # #         try:
# # # # #             person_count = detector_engine.detect_persons(file_path)
# # # # #         except:
# # # # #             pass
        
# # # # #         img_record = DBImage(
# # # # #             filename=filename, original_path=file_path, timestamp=datetime.now(),
# # # # #             ocr_text=ocr_text, person_count=person_count, width=width,
# # # # #             avg_r=avg_r, avg_g=avg_g, avg_b=avg_b, height=height
# # # # #         )
# # # # #         db.add(img_record)
# # # # #         db.flush()
        
# # # # #         if clip_emb is not None:
# # # # #             try:
# # # # #                 if search_engine.index is None:
# # # # #                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
# # # # #                 new_vec = clip_emb.reshape(1, -1).astype('float32')
# # # # #                 faiss.normalize_L2(new_vec)
# # # # #                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id]).astype('int64'))
# # # # #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# # # # #             except:
# # # # #                 pass
        
# # # # #         db.commit()

# # # # #         return {"status": "success", "id": img_record.id, "filename": filename}
# # # # #     except Exception as e:
# # # # #         db.rollback()
# # # # #         if os.path.exists(file_path):
# # # # #             os.remove(file_path)
# # # # #         raise HTTPException(status_code=500)
# # # # #     finally:
# # # # #         db.close()

# # # # # if __name__ == "__main__":
# # # # #     import uvicorn
# # # # #     uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# # # # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from fastapi.staticfiles import StaticFiles
# # # # import os, uuid, shutil, numpy as np, faiss
# # # # from datetime import datetime
# # # # import logging
# # # # from contextlib import asynccontextmanager
# # # # import datetime as datetime_module

# # # # logging.basicConfig(level=logging.INFO)
# # # # logger = logging.getLogger("main")

# # # # from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
# # # # from search_engine import search_engine, resolve_query
# # # # from voice_engine import voice_engine
# # # # from face_engine import face_engine
# # # # from ocr_engine import extract_text
# # # # from detector_engine import detector_engine
# # # # from duplicate_engine import duplicate_engine
# # # # from clustering_engine import clustering_engine

# # # # IMAGE_DIR = "../data/images"
# # # # FAISS_INDEX_PATH = "../data/index.faiss"

# # # # FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
# # # # FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
# # # # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # # # RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # # # RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # # # CLIP_SCORE_MIN = 0.10

# # # # RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # # # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # # # recluster_last_triggered = None

# # # # def should_trigger_recluster(background_tasks):
# # # #     global recluster_last_triggered
# # # #     if not RECLUSTER_ON_UPLOAD or not background_tasks:
# # # #         return
# # # #     try:
# # # #         counter = 0
# # # #         if os.path.exists(RECLUSTER_COUNTER_PATH):
# # # #             try:
# # # #                 with open(RECLUSTER_COUNTER_PATH, 'r') as f:
# # # #                     counter = int(f.read().strip())
# # # #             except:
# # # #                 pass
# # # #         counter += 1
# # # #         with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # #             f.write(str(counter))
# # # #         should_trigger = counter >= RECLUSTER_BATCH_SIZE
# # # #         now = datetime_module.datetime.now()
# # # #         if recluster_last_triggered:
# # # #             elapsed = (now - recluster_last_triggered).total_seconds()
# # # #             if elapsed >= RECLUSTER_TIMER_SECONDS:
# # # #                 should_trigger = True
# # # #         elif counter > 0:
# # # #             should_trigger = counter >= RECLUSTER_BATCH_SIZE
# # # #         if should_trigger:
# # # #             logger.info(f"📊 Recluster triggered")
# # # #             background_tasks.add_task(recluster)
# # # #             recluster_last_triggered = now
# # # #             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # # #                 f.write('0')
# # # #     except Exception as e:
# # # #         logger.warning(f"Recluster check failed: {e}")

# # # # @asynccontextmanager
# # # # async def lifespan(app: FastAPI):
# # # #     init_db()
# # # #     logger.info("🚀 Starting with SMART TWO-PASS search...")
# # # #     if os.path.exists(FAISS_INDEX_PATH):
# # # #         try:
# # # #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# # # #             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
# # # #         except Exception as e:
# # # #             logger.error(f"Index load failed: {e}")
# # # #             search_engine.index = None
# # # #     logger.info("✅ Ready!")
# # # #     yield

# # # # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)
# # # # app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# # # # if not os.path.exists(IMAGE_DIR):
# # # #     os.makedirs(IMAGE_DIR)
# # # # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # # # @app.get("/health")
# # # # def health():
# # # #     return {"status": "ready", "mode": "SMART_TWO_PASS", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# # # # @app.get("/test-db")
# # # # def test_db():
# # # #     db = SessionLocal()
# # # #     try:
# # # #         count = db.query(DBImage).count()
# # # #         images = db.query(DBImage).limit(1).all()
# # # #         return {"status": "ok", "total_images": count, "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None}
# # # #     except Exception as e:
# # # #         return {"status": "error", "message": str(e)}
# # # #     finally:
# # # #         db.close()

# # # # @app.post("/search")
# # # # def search(query: str = Form(...), top_k: int = Form(20)):
# # # #     """
# # # #     SMART TWO-PASS SEARCH:
# # # #     Pass 1: Find keyword-confirmed images, get minimum CLIP score
# # # #     Pass 2: Use that minimum as threshold for ALL images
# # # #     Result: ONLY relevant images shown!
# # # #     """
# # # #     if not query or not query.strip():
# # # #         return {"status": "error", "message": "Query empty"}

# # # #     processed_query = resolve_query(query)
# # # #     logger.info(f"🔍 Search: '{query}'")

# # # #     COLOR_MAP = {
# # # #         'red': (1.0, 0, 0), 'blue': (0, 0, 1.0), 'green': (0, 1.0, 0),
# # # #         'yellow': (1.0, 1.0, 0), 'orange': (1.0, 0.5, 0), 'purple': (0.5, 0, 0.5),
# # # #         'pink': (1.0, 0.75, 0.8), 'black': (0, 0, 0), 'white': (1, 1, 1),
# # # #         'gray': (0.5, 0.5, 0.5), 'brown': (0.6, 0.4, 0.2)
# # # #     }
# # # #     query_lower = query.lower()
# # # #     query_colors = [rgb for name, rgb in COLOR_MAP.items() if name in query_lower]

# # # #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# # # #     if query_emb is None or search_engine.index is None:
# # # #         return {"status": "error", "message": "No images indexed"}

# # # #     candidate_k = min(top_k * 8, 250)
# # # #     query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
# # # #     faiss.normalize_L2(query_emb_reshaped)
# # # #     distances, indices = search_engine.index.search(query_emb_reshaped, candidate_k)

# # # #     db = SessionLocal()
# # # #     try:
# # # #         # ===== PASS 1: Collect ALL candidates with bonuses =====
# # # #         all_candidates = []

# # # #         for dist, idx in zip(distances[0], indices[0]):
# # # #             if idx == -1:
# # # #                 continue
# # # #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# # # #             if not img:
# # # #                 continue

# # # #             raw_sim = float(dist)
# # # #             clip_score = max(0.0, raw_sim)
            
# # # #             if clip_score < CLIP_SCORE_MIN:
# # # #                 continue

# # # #             # OCR matching
# # # #             ocr_text = (img.ocr_text or "").lower()
# # # #             query_words = processed_query.lower().split()
# # # #             significant_words = [w for w in query_words if len(w) > 2]
            
# # # #             ocr_bonus = 0.0
# # # #             if significant_words and ocr_text:
# # # #                 matches = sum(1 for w in significant_words if w in ocr_text)
# # # #                 if matches > 0:
# # # #                     ocr_bonus = min(matches / len(significant_words), 1.0)

# # # #             # Tag matching
# # # #             tag_bonus = 0.0
# # # #             if img.scene_label:
# # # #                 tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
# # # #                 query_objects = [w for w in query_words if len(w) > 2]
# # # #                 if query_objects:
# # # #                     for obj in query_objects:
# # # #                         if any(obj in tag for tag in tags):
# # # #                             tag_bonus = 1.0
# # # #                             break

# # # #             # Color matching
# # # #             color_bonus = 0.0
# # # #             if query_colors and getattr(img, 'avg_r', None) is not None:
# # # #                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# # # #                 for qc in query_colors:
# # # #                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# # # #                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
# # # #                     color_bonus = max(color_bonus, score)

# # # #             has_keyword_match = ocr_bonus > 0 or tag_bonus > 0
            
# # # #             if has_keyword_match:
# # # #                 final_score = (
# # # #                     (0.40 * clip_score) +
# # # #                     (0.35 * ocr_bonus) +
# # # #                     (0.15 * color_bonus) +
# # # #                     (0.10 * tag_bonus)
# # # #                 )
# # # #             else:
# # # #                 final_score = clip_score

# # # #             all_candidates.append({
# # # #                 'img': img,
# # # #                 'clip_score': clip_score,
# # # #                 'final_score': final_score,
# # # #                 'ocr_bonus': ocr_bonus,
# # # #                 'tag_bonus': tag_bonus,
# # # #                 'color_bonus': color_bonus,
# # # #                 'has_keyword_match': has_keyword_match
# # # #             })

# # # #         # ===== CALCULATE THRESHOLD =====
# # # #         keyword_confirmed_clips = [c['clip_score'] for c in all_candidates if c['has_keyword_match']]

# # # #         if keyword_confirmed_clips:
# # # #             # Use minimum keyword-confirmed as threshold
# # # #             threshold = min(keyword_confirmed_clips)
# # # #             logger.info(f"🎯 Threshold from {len(keyword_confirmed_clips)} confirmed: {threshold:.4f}")
# # # #         else:
# # # #             # No keywords - use adaptive fallback
# # # #             all_clips = [c['clip_score'] for c in all_candidates]
# # # #             if all_clips:
# # # #                 best_clip = max(all_clips)
# # # #                 threshold = best_clip - 0.05
# # # #                 logger.info(f"⚠️ Adaptive threshold: {threshold:.4f}")
# # # #             else:
# # # #                 return {"status": "not_found", "message": f"No images for '{query}'"}

# # # #         # ===== PASS 2: Filter by threshold =====
# # # #         results = []

# # # #         for candidate in all_candidates:
# # # #             img = candidate['img']
# # # #             clip_score = candidate['clip_score']
# # # #             final_score = candidate['final_score']
# # # #             ocr_bonus = candidate['ocr_bonus']
# # # #             tag_bonus = candidate['tag_bonus']
# # # #             has_keyword_match = candidate['has_keyword_match']
            
# # # #             # Check threshold
# # # #             if has_keyword_match:
# # # #                 show = True
# # # #             else:
# # # #                 show = clip_score >= threshold
            
# # # #             if not show:
# # # #                 logger.debug(f"❌ Skip: CLIP {clip_score:.4f} < {threshold:.4f}")
# # # #                 continue
            
# # # #             logger.debug(f"✅ {img.filename}: CLIP={clip_score:.4f}, FINAL={final_score:.4f}")
            
# # # #             results.append({
# # # #                 "id": img.id,
# # # #                 "filename": img.filename,
# # # #                 "score": round(final_score * 100, 2),
# # # #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # # #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # # #                 "person_count": img.person_count or 0
# # # #             })

# # # #         results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
# # # #         if not results:
# # # #             return {"status": "not_found", "message": f"No match for '{query}'"}

# # # #         logger.info(f"✅ Found {len(results)} results")
# # # #         return {"status": "found", "query": query, "count": len(results), "threshold": round(threshold * 100, 1), "results": results}
# # # #     finally:
# # # #         db.close()

# # # # @app.post("/search/voice")
# # # # def voice_search(duration: int = Form(5)):
# # # #     try:
# # # #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# # # #         if not transcribed:
# # # #             return {"status": "error", "message": "No audio"}
# # # #         return search(query=transcribed, top_k=20)
# # # #     except Exception as e:
# # # #         return {"status": "error", "message": str(e)}

# # # # @app.get("/timeline")
# # # # def get_timeline():
# # # #     db = SessionLocal()
# # # #     try:
# # # #         images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
# # # #         results = [{"id": img.id, "filename": img.filename, "date": img.timestamp.isoformat() if img.timestamp else None, "thumbnail": f"/images/{img.filename}"} for img in images]
# # # #         return {"count": len(results), "results": results}
# # # #     finally:
# # # #         db.close()

# # # # @app.get("/faces")
# # # # def get_faces(person_id: int = Query(None)):
# # # #     db = SessionLocal()
# # # #     try:
# # # #         if person_id:
# # # #             # ── Detail view for one person ──────────────────────────────────
# # # #             person = db.query(Person).filter(Person.id == person_id).first()
# # # #             if not person:
# # # #                 raise HTTPException(status_code=404)
# # # #             faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# # # #             images = []
# # # #             cover = None
# # # #             for f in faces:
# # # #                 img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# # # #                 if img:
# # # #                     if cover is None:
# # # #                         cover = f"/images/{img.filename}"
# # # #                     images.append({
# # # #                         "id": img.id,
# # # #                         "filename": img.filename,
# # # #                         "thumbnail": f"/images/{img.filename}",
# # # #                         "date": img.timestamp.isoformat() if img.timestamp else None,
# # # #                     })
# # # #             return {
# # # #                 "id": person.id,
# # # #                 "name": person.name,
# # # #                 "face_count": len(faces),
# # # #                 "cover": cover,
# # # #                 "images": images,
# # # #             }
# # # #         else:
# # # #             # ── List all people with cover photo ────────────────────────────
# # # #             people = db.query(Person).all()
# # # #             results = []
# # # #             for p in people:
# # # #                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# # # #                 if not faces:
# # # #                     continue
# # # #                 # Find a cover image (first face that has a valid image file)
# # # #                 cover = None
# # # #                 for f in faces:
# # # #                     img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# # # #                     if img and img.filename:
# # # #                         cover = f"/images/{img.filename}"
# # # #                         break
# # # #                 results.append({
# # # #                     "id": p.id,
# # # #                     "name": p.name,
# # # #                     "count": len(faces),
# # # #                     "cover": cover,
# # # #                 })
# # # #             return {"results": results, "count": len(results)}
# # # #     finally:
# # # #         db.close()

# # # # @app.post("/people/{person_id}")
# # # # def update_person(person_id: int, name: str = Form(...)):
# # # #     db = SessionLocal()
# # # #     try:
# # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # #         if not person:
# # # #             raise HTTPException(status_code=404)
# # # #         person.name = name
# # # #         db.commit()
# # # #         return {"status": "success", "id": person.id, "name": person.name}
# # # #     finally:
# # # #         db.close()

# # # # @app.get("/people/{person_id}/celebcheck")
# # # # def check_celebrity_match(person_id: int):
# # # #     db = SessionLocal()
# # # #     try:
# # # #         person = db.query(Person).filter(Person.id == person_id).first()
# # # #         if not person:
# # # #             return {"status": "no_match"}
# # # #         return {"status": "no_match"}
# # # #     finally:
# # # #         db.close()

# # # # @app.get("/albums")
# # # # def get_albums(album_id: int = Query(None)):
# # # #     db = SessionLocal()
# # # #     try:
# # # #         if album_id:
# # # #             # ── Detail view for one album ────────────────────────────────────
# # # #             album = db.query(Album).filter(Album.id == album_id).first()
# # # #             if not album:
# # # #                 raise HTTPException(status_code=404)
# # # #             images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
# # # #             cover = f"/images/{images[0].filename}" if images else None
# # # #             date_str = ""
# # # #             if album.start_date:
# # # #                 date_str = album.start_date.strftime("%b %Y")
# # # #                 if album.end_date and album.end_date.month != album.start_date.month:
# # # #                     date_str += f" – {album.end_date.strftime('%b %Y')}"
# # # #             return {
# # # #                 "id": album.id,
# # # #                 "title": album.title,
# # # #                 "type": album.type,
# # # #                 "description": album.description,
# # # #                 "date": date_str,
# # # #                 "start_date": album.start_date.isoformat() if album.start_date else None,
# # # #                 "end_date": album.end_date.isoformat() if album.end_date else None,
# # # #                 "cover": cover,
# # # #                 "image_count": len(images),
# # # #                 "images": [
# # # #                     {
# # # #                         "id": img.id,
# # # #                         "filename": img.filename,
# # # #                         "thumbnail": f"/images/{img.filename}",
# # # #                         "date": img.timestamp.isoformat() if img.timestamp else None,
# # # #                     }
# # # #                     for img in images
# # # #                 ],
# # # #             }
# # # #         else:
# # # #             # ── List all albums ──────────────────────────────────────────────
# # # #             albums = db.query(Album).all()
# # # #             results = []
# # # #             for a in albums:
# # # #                 album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
# # # #                 cover = f"/images/{album_images[0].filename}" if album_images else None
# # # #                 date_str = ""
# # # #                 if a.start_date:
# # # #                     date_str = a.start_date.strftime("%b %Y")
# # # #                     if a.end_date and a.end_date.month != a.start_date.month:
# # # #                         date_str += f" – {a.end_date.strftime('%b %Y')}"
# # # #                 results.append({
# # # #                     "id": a.id,
# # # #                     "title": a.title,
# # # #                     "type": a.type,
# # # #                     "description": a.description,
# # # #                     "date": date_str,
# # # #                     "cover": cover,
# # # #                     "count": len(album_images),
# # # #                     "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]],
# # # #                 })
# # # #             return {"results": results, "count": len(results)}
# # # #     finally:
# # # #         db.close()

# # # # @app.post("/favorites")
# # # # def add_favorite(image_id: int = Form(...)):
# # # #     db = SessionLocal()
# # # #     try:
# # # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # # #         if not img:
# # # #             raise HTTPException(status_code=404)
# # # #         img.is_favorite = not getattr(img, 'is_favorite', False)
# # # #         db.commit()
# # # #         return {"status": "success"}
# # # #     finally:
# # # #         db.close()

# # # # @app.get("/favorites")
# # # # def get_favorites():
# # # #     db = SessionLocal()
# # # #     try:
# # # #         images = db.query(DBImage).filter(DBImage.is_favorite == True).all()
# # # #         return {"count": len(images), "results": [{"id": img.id, "filename": img.filename} for img in images]}
# # # #     finally:
# # # #         db.close()

# # # # @app.get("/duplicates")
# # # # def get_duplicates():
# # # #     db = SessionLocal()
# # # #     try:
# # # #         all_images = db.query(DBImage).all()
# # # #         return {"status": "found", "duplicate_groups": [], "total_groups": 0}
# # # #     finally:
# # # #         db.close()

# # # # @app.post("/recluster")
# # # # def recluster():
# # # #     """
# # # #     Wipe all person / album assignments and re-run face clustering + event detection
# # # #     from scratch using the face embeddings already stored in the DB.

# # # #     Old code only CLEARED data and returned zeros — it never ran any clustering.
# # # #     """
# # # #     db = SessionLocal()
# # # #     try:
# # # #         logger.info("🔄 Recluster: clearing old assignments…")

# # # #         # ── 1. Clear previous clustering results ────────────────────────────
# # # #         db.query(DBFace).update({"person_id": None})
# # # #         db.query(Person).delete()
# # # #         db.query(DBImage).update({"album_id": None})
# # # #         db.query(Album).filter(Album.type == "event").delete()
# # # #         db.commit()

# # # #         # ── 2. Load all face embeddings from the DB ──────────────────────────
# # # #         face_records = db.query(DBFace).filter(DBFace.face_embedding != None).all()
# # # #         embeddings = []
# # # #         valid_face_records = []

# # # #         for fr in face_records:
# # # #             try:
# # # #                 emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
# # # #                 if emb.shape[0] == 512:          # ArcFace dimension
# # # #                     embeddings.append(emb)
# # # #                     valid_face_records.append(fr)
# # # #             except Exception as e:
# # # #                 logger.warning(f"Bad embedding for face {fr.id}: {e}")

# # # #         people_count = 0
# # # #         if embeddings:
# # # #             logger.info(f"👥 Clustering {len(embeddings)} face embeddings…")
# # # #             labels = face_engine.cluster_faces(embeddings)

# # # #             person_map = {}   # cluster label → Person.id
# # # #             for i, label in enumerate(labels):
# # # #                 if label == -1:
# # # #                     continue   # noise — left unassigned
# # # #                 if label not in person_map:
# # # #                     new_person = Person(name=f"Person {label + 1}")
# # # #                     db.add(new_person)
# # # #                     db.flush()
# # # #                     person_map[label] = new_person.id
# # # #                     people_count += 1
# # # #                 valid_face_records[i].person_id = person_map[label]

# # # #             db.commit()
# # # #             logger.info(f"✅ Clustered into {people_count} people")
# # # #         else:
# # # #             logger.warning("⚠️  No face embeddings found — run build_index.py first")

# # # #         # ── 3. Album / event detection ───────────────────────────────────────
# # # #         all_images = db.query(DBImage).all()
# # # #         albums_count = 0

# # # #         if all_images:
# # # #             metadata = [
# # # #                 {
# # # #                     "id": img.id,
# # # #                     "lat": img.lat or 0.0,
# # # #                     "lon": img.lon or 0.0,
# # # #                     "timestamp": img.timestamp,
# # # #                 }
# # # #                 for img in all_images if img.timestamp
# # # #             ]

# # # #             if metadata:
# # # #                 album_labels = clustering_engine.detect_events(metadata)
# # # #                 album_map = {}

# # # #                 for i, label in enumerate(album_labels):
# # # #                     if label == -1:
# # # #                         continue
# # # #                     if label not in album_map:
# # # #                         cluster_imgs = [metadata[j] for j, l in enumerate(album_labels) if l == label]
# # # #                         ts_list = [m["timestamp"] for m in cluster_imgs if m["timestamp"]]
# # # #                         start_d = min(ts_list) if ts_list else None
# # # #                         end_d   = max(ts_list) if ts_list else None

# # # #                         title = f"Event {label + 1}"
# # # #                         if start_d:
# # # #                             title = start_d.strftime("%b %Y")

# # # #                         new_album = Album(
# # # #                             title=title,
# # # #                             type="event",
# # # #                             start_date=start_d,
# # # #                             end_date=end_d,
# # # #                         )
# # # #                         db.add(new_album)
# # # #                         db.flush()
# # # #                         album_map[label] = new_album.id
# # # #                         albums_count += 1

# # # #                     # assign the image (use original all_images ordering by index)
# # # #                     # metadata was built in the same order as all_images (filtered to those with timestamps)
# # # #                     img_id = metadata[i]["id"]
# # # #                     db.query(DBImage).filter(DBImage.id == img_id).update({"album_id": album_map[label]})

# # # #                 db.commit()
# # # #                 logger.info(f"✅ Created {albums_count} albums/events")

# # # #         return {"status": "done", "people": people_count, "albums": albums_count}
# # # #     except Exception as e:
# # # #         db.rollback()
# # # #         logger.error(f"❌ Recluster failed: {e}", exc_info=True)
# # # #         raise HTTPException(status_code=500, detail=str(e))
# # # #     finally:
# # # #         db.close()

# # # # @app.post("/upload")
# # # # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# # # #     ext = os.path.splitext(file.filename)[1].lower()
# # # #     if ext not in [".jpg", ".jpeg", ".png"]:
# # # #         raise HTTPException(status_code=400)
    
# # # #     filename = f"{uuid.uuid4()}{ext}"
# # # #     file_path = os.path.join(IMAGE_DIR, filename)
# # # #     db = SessionLocal()
    
# # # #     try:
# # # #         with open(file_path, "wb") as buffer:
# # # #             shutil.copyfileobj(file.file, buffer)
        
# # # #         from PIL import Image as PILImage
# # # #         import json as _json
# # # #         width = height = None
# # # #         avg_r = avg_g = avg_b = 0.0
# # # #         try:
# # # #             img_pil = PILImage.open(file_path).convert("RGB")
# # # #             width, height = img_pil.size
# # # #             import numpy as _np
# # # #             arr = _np.array(img_pil)
# # # #             avg_r = float(arr[:, :, 0].mean())
# # # #             avg_g = float(arr[:, :, 1].mean())
# # # #             avg_b = float(arr[:, :, 2].mean())
# # # #         except Exception:
# # # #             pass
        
# # # #         # CLIP embedding
# # # #         clip_emb = None
# # # #         try:
# # # #             clip_emb = search_engine.get_image_embedding(file_path)
# # # #         except Exception:
# # # #             pass

# # # #         # OCR
# # # #         ocr_text = ""
# # # #         try:
# # # #             ocr_text = extract_text(file_path)
# # # #         except Exception:
# # # #             pass
        
# # # #         # Object / scene detection → scene_label
# # # #         scene_label = ""
# # # #         person_count = 0
# # # #         try:
# # # #             objects = detector_engine.detect_objects(file_path, threshold=0.5)
# # # #             scene_label = ", ".join(objects) if objects else ""
# # # #             person_count = objects.count("person") if objects else detector_engine.detect_persons(file_path)
# # # #         except Exception:
# # # #             try:
# # # #                 person_count = detector_engine.detect_persons(file_path)
# # # #             except Exception:
# # # #                 pass
        
# # # #         img_record = DBImage(
# # # #             filename=filename,
# # # #             original_path=file_path,
# # # #             timestamp=datetime.now(),
# # # #             ocr_text=ocr_text,
# # # #             scene_label=scene_label,
# # # #             person_count=person_count,
# # # #             width=width,
# # # #             height=height,
# # # #             avg_r=avg_r,
# # # #             avg_g=avg_g,
# # # #             avg_b=avg_b,
# # # #         )
# # # #         db.add(img_record)
# # # #         db.flush()
        
# # # #         # CLIP → FAISS index
# # # #         if clip_emb is not None:
# # # #             try:
# # # #                 if search_engine.index is None:
# # # #                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
# # # #                 new_vec = clip_emb.reshape(1, -1).astype('float32')
# # # #                 faiss.normalize_L2(new_vec)
# # # #                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id]).astype('int64'))
# # # #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# # # #             except Exception as e:
# # # #                 logger.warning(f"FAISS update failed: {e}")
        
# # # #         # Face detection → store embeddings in DB so /recluster can use them
# # # #         face_count = 0
# # # #         try:
# # # #             faces = face_engine.detect_faces(file_path)
# # # #             for face in faces:
# # # #                 emb = face["embedding"].astype(np.float32)
# # # #                 face_record = DBFace(
# # # #                     image_id=img_record.id,
# # # #                     bbox=_json.dumps(face["bbox"]),
# # # #                     face_embedding=emb.tobytes(),
# # # #                 )
# # # #                 db.add(face_record)
# # # #                 face_count += 1
# # # #         except Exception as e:
# # # #             logger.warning(f"Face detection failed for upload: {e}")
        
# # # #         db.commit()

# # # #         if background_tasks is not None:
# # # #             should_trigger_recluster(background_tasks)

# # # #         return {
# # # #             "status": "success",
# # # #             "id": img_record.id,
# # # #             "filename": filename,
# # # #             "person_count": person_count,
# # # #             "face_count": face_count,
# # # #         }
# # # #     except Exception as e:
# # # #         db.rollback()
# # # #         if os.path.exists(file_path):
# # # #             os.remove(file_path)
# # # #         logger.error(f"Upload failed: {e}", exc_info=True)
# # # #         raise HTTPException(status_code=500)
# # # #     finally:
# # # #         db.close()

# # # # if __name__ == "__main__":
# # # #     import uvicorn
# # # #     uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# # # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.staticfiles import StaticFiles
# # # import os, uuid, shutil, numpy as np, faiss
# # # from datetime import datetime
# # # import logging
# # # from contextlib import asynccontextmanager
# # # import datetime as datetime_module

# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger("main")

# # # from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
# # # from search_engine import search_engine, resolve_query
# # # from voice_engine import voice_engine
# # # from face_engine import face_engine
# # # from ocr_engine import extract_text
# # # from detector_engine import detector_engine
# # # from duplicate_engine import duplicate_engine
# # # from clustering_engine import clustering_engine

# # # IMAGE_DIR = "../data/images"
# # # FAISS_INDEX_PATH = "../data/index.faiss"

# # # FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
# # # FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
# # # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # # RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # # RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # # CLIP_SCORE_MIN = 0.10

# # # RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # # recluster_last_triggered = None

# # # def should_trigger_recluster(background_tasks):
# # #     global recluster_last_triggered
# # #     if not RECLUSTER_ON_UPLOAD or not background_tasks:
# # #         return
# # #     try:
# # #         counter = 0
# # #         if os.path.exists(RECLUSTER_COUNTER_PATH):
# # #             try:
# # #                 with open(RECLUSTER_COUNTER_PATH, 'r') as f:
# # #                     counter = int(f.read().strip())
# # #             except:
# # #                 pass
# # #         counter += 1
# # #         with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # #             f.write(str(counter))
# # #         should_trigger = counter >= RECLUSTER_BATCH_SIZE
# # #         now = datetime_module.datetime.now()
# # #         if recluster_last_triggered:
# # #             elapsed = (now - recluster_last_triggered).total_seconds()
# # #             if elapsed >= RECLUSTER_TIMER_SECONDS:
# # #                 should_trigger = True
# # #         elif counter > 0:
# # #             should_trigger = counter >= RECLUSTER_BATCH_SIZE
# # #         if should_trigger:
# # #             logger.info(f"📊 Recluster triggered")
# # #             background_tasks.add_task(recluster)
# # #             recluster_last_triggered = now
# # #             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # #                 f.write('0')
# # #     except Exception as e:
# # #         logger.warning(f"Recluster check failed: {e}")

# # # @asynccontextmanager
# # # async def lifespan(app: FastAPI):
# # #     init_db()
# # #     logger.info("🚀 Starting with SMART TWO-PASS search...")
# # #     if os.path.exists(FAISS_INDEX_PATH):
# # #         try:
# # #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# # #             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
# # #         except Exception as e:
# # #             logger.error(f"Index load failed: {e}")
# # #             search_engine.index = None
# # #     logger.info("✅ Ready!")
# # #     yield

# # # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)
# # # app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# # # if not os.path.exists(IMAGE_DIR):
# # #     os.makedirs(IMAGE_DIR)
# # # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # # @app.get("/health")
# # # def health():
# # #     return {"status": "ready", "mode": "SMART_TWO_PASS", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# # # @app.get("/test-db")
# # # def test_db():
# # #     db = SessionLocal()
# # #     try:
# # #         count = db.query(DBImage).count()
# # #         images = db.query(DBImage).limit(1).all()
# # #         return {"status": "ok", "total_images": count, "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None}
# # #     except Exception as e:
# # #         return {"status": "error", "message": str(e)}
# # #     finally:
# # #         db.close()

# # # @app.post("/search")
# # # def search(query: str = Form(...), top_k: int = Form(20)):
# # #     """
# # #     SMART TWO-PASS SEARCH:
# # #     Pass 1: Find keyword-confirmed images, get minimum CLIP score
# # #     Pass 2: Use that minimum as threshold for ALL images
# # #     Result: ONLY relevant images shown!
# # #     """
# # #     if not query or not query.strip():
# # #         return {"status": "error", "message": "Query empty"}

# # #     processed_query = resolve_query(query)
# # #     logger.info(f"🔍 Search: '{query}'")

# # #     COLOR_MAP = {
# # #         'red': (1.0, 0, 0), 'blue': (0, 0, 1.0), 'green': (0, 1.0, 0),
# # #         'yellow': (1.0, 1.0, 0), 'orange': (1.0, 0.5, 0), 'purple': (0.5, 0, 0.5),
# # #         'pink': (1.0, 0.75, 0.8), 'black': (0, 0, 0), 'white': (1, 1, 1),
# # #         'gray': (0.5, 0.5, 0.5), 'brown': (0.6, 0.4, 0.2)
# # #     }
# # #     query_lower = query.lower()
# # #     query_colors = [rgb for name, rgb in COLOR_MAP.items() if name in query_lower]

# # #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# # #     if query_emb is None or search_engine.index is None:
# # #         return {"status": "error", "message": "No images indexed"}

# # #     candidate_k = min(top_k * 8, 250)
# # #     query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
# # #     faiss.normalize_L2(query_emb_reshaped)
# # #     distances, indices = search_engine.index.search(query_emb_reshaped, candidate_k)

# # #     db = SessionLocal()
# # #     try:
# # #         # ===== PASS 1: Collect ALL candidates with bonuses =====
# # #         all_candidates = []

# # #         for dist, idx in zip(distances[0], indices[0]):
# # #             if idx == -1:
# # #                 continue
# # #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# # #             if not img:
# # #                 continue

# # #             raw_sim = float(dist)
# # #             clip_score = max(0.0, raw_sim)
            
# # #             if clip_score < CLIP_SCORE_MIN:
# # #                 continue

# # #             # OCR matching
# # #             ocr_text = (img.ocr_text or "").lower()
# # #             query_words = processed_query.lower().split()
# # #             significant_words = [w for w in query_words if len(w) > 2]
            
# # #             ocr_bonus = 0.0
# # #             if significant_words and ocr_text:
# # #                 matches = sum(1 for w in significant_words if w in ocr_text)
# # #                 if matches > 0:
# # #                     ocr_bonus = min(matches / len(significant_words), 1.0)

# # #             # Tag matching — exact word only (substring caused "man"→"woman", "dog"→"hotdog")
# # #             tag_bonus = 0.0
# # #             if img.scene_label:
# # #                 tag_words = set()
# # #                 for tag in img.scene_label.split(","):
# # #                     for word in tag.strip().lower().split():
# # #                         tag_words.add(word)
# # #                 query_objects = [w for w in query_words if len(w) > 2]
# # #                 if any(obj in tag_words for obj in query_objects):
# # #                     tag_bonus = 1.0

# # #             # Color matching
# # #             color_bonus = 0.0
# # #             if query_colors and getattr(img, 'avg_r', None) is not None:
# # #                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# # #                 for qc in query_colors:
# # #                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# # #                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
# # #                     color_bonus = max(color_bonus, score)

# # #             has_keyword_match = ocr_bonus > 0 or tag_bonus > 0
            
# # #             if has_keyword_match:
# # #                 final_score = (
# # #                     (0.40 * clip_score) +
# # #                     (0.35 * ocr_bonus) +
# # #                     (0.15 * color_bonus) +
# # #                     (0.10 * tag_bonus)
# # #                 )
# # #             else:
# # #                 final_score = clip_score

# # #             all_candidates.append({
# # #                 'img': img,
# # #                 'clip_score': clip_score,
# # #                 'final_score': final_score,
# # #                 'ocr_bonus': ocr_bonus,
# # #                 'tag_bonus': tag_bonus,
# # #                 'color_bonus': color_bonus,
# # #                 'has_keyword_match': has_keyword_match
# # #             })

# # #         # ===== CALCULATE THRESHOLD =====
# # #         keyword_confirmed_clips = [c['clip_score'] for c in all_candidates if c['has_keyword_match']]

# # #         if keyword_confirmed_clips:
# # #             threshold = max(min(keyword_confirmed_clips), THRESHOLD_FLOOR)
# # #             logger.info(f"🎯 Threshold from {len(keyword_confirmed_clips)} confirmed: {threshold:.4f}")
# # #         else:
# # #             all_clips = [c['clip_score'] for c in all_candidates]
# # #             if all_clips:
# # #                 best_clip = max(all_clips)
# # #                 threshold = max(best_clip * ADAPTIVE_RATIO, THRESHOLD_FLOOR)
# # #                 logger.info(f"⚠️  Adaptive threshold: {threshold:.4f}")
# # #             else:
# # #                 return {"status": "not_found", "message": f"No images for '{query}'"}

# # #         # ===== PASS 2: Filter by threshold =====
# # #         results = []

# # #         for candidate in all_candidates:
# # #             img = candidate['img']
# # #             clip_score = candidate['clip_score']
# # #             final_score = candidate['final_score']
# # #             has_keyword_match = candidate['has_keyword_match']

# # #             # Every image must pass the threshold — keyword match only affects score weight.
# # #             if clip_score < threshold:
# # #                 logger.debug(f"❌ Skip {img.filename}: CLIP {clip_score:.4f} < {threshold:.4f}")
# # #                 continue
            
# # #             logger.debug(f"✅ {img.filename}: CLIP={clip_score:.4f}, FINAL={final_score:.4f}")
            
# # #             results.append({
# # #                 "id": img.id,
# # #                 "filename": img.filename,
# # #                 "score": round(final_score * 100, 2),
# # #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # #                 "person_count": img.person_count or 0
# # #             })

# # #         results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
# # #         if not results:
# # #             return {"status": "not_found", "message": f"No match for '{query}'"}

# # #         logger.info(f"✅ Found {len(results)} results")
# # #         return {"status": "found", "query": query, "count": len(results), "threshold": round(threshold * 100, 1), "results": results}
# # #     finally:
# # #         db.close()

# # # @app.post("/search/voice")
# # # def voice_search(duration: int = Form(5)):
# # #     try:
# # #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# # #         if not transcribed:
# # #             return {"status": "error", "message": "No audio"}
# # #         return search(query=transcribed, top_k=20)
# # #     except Exception as e:
# # #         return {"status": "error", "message": str(e)}

# # # @app.get("/timeline")
# # # def get_timeline():
# # #     db = SessionLocal()
# # #     try:
# # #         images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
# # #         results = [{"id": img.id, "filename": img.filename, "date": img.timestamp.isoformat() if img.timestamp else None, "thumbnail": f"/images/{img.filename}"} for img in images]
# # #         return {"count": len(results), "results": results}
# # #     finally:
# # #         db.close()

# # # @app.get("/faces")
# # # def get_faces(person_id: int = Query(None)):
# # #     db = SessionLocal()
# # #     try:
# # #         if person_id:
# # #             person = db.query(Person).filter(Person.id == person_id).first()
# # #             if not person:
# # #                 raise HTTPException(status_code=404)
# # #             faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# # #             images = []
# # #             cover = None
# # #             for f in faces:
# # #                 img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# # #                 if img:
# # #                     if cover is None:
# # #                         cover = f"/images/{img.filename}"
# # #                     images.append({
# # #                         "id": img.id,
# # #                         "filename": img.filename,
# # #                         "thumbnail": f"/images/{img.filename}",
# # #                         "date": img.timestamp.isoformat() if img.timestamp else None,
# # #                     })
# # #             return {"id": person.id, "name": person.name, "face_count": len(faces), "cover": cover, "images": images}
# # #         else:
# # #             people = db.query(Person).all()
# # #             results = []
# # #             for p in people:
# # #                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# # #                 if not faces:
# # #                     continue
# # #                 cover = None
# # #                 for f in faces:
# # #                     img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# # #                     if img and img.filename:
# # #                         cover = f"/images/{img.filename}"
# # #                         break
# # #                 results.append({"id": p.id, "name": p.name, "count": len(faces), "cover": cover})
# # #             return {"results": results, "count": len(results)}
# # #     finally:
# # #         db.close()

# # # @app.post("/people/{person_id}")
# # # def update_person(person_id: int, name: str = Form(...)):
# # #     db = SessionLocal()
# # #     try:
# # #         person = db.query(Person).filter(Person.id == person_id).first()
# # #         if not person:
# # #             raise HTTPException(status_code=404)
# # #         person.name = name
# # #         db.commit()
# # #         return {"status": "success", "id": person.id, "name": person.name}
# # #     finally:
# # #         db.close()

# # # @app.get("/people/{person_id}/celebcheck")
# # # def check_celebrity_match(person_id: int):
# # #     db = SessionLocal()
# # #     try:
# # #         person = db.query(Person).filter(Person.id == person_id).first()
# # #         if not person:
# # #             return {"status": "no_match"}
# # #         return {"status": "no_match"}
# # #     finally:
# # #         db.close()

# # # @app.get("/albums")
# # # def get_albums(album_id: int = Query(None)):
# # #     db = SessionLocal()
# # #     try:
# # #         if album_id:
# # #             album = db.query(Album).filter(Album.id == album_id).first()
# # #             if not album:
# # #                 raise HTTPException(status_code=404)
# # #             images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
# # #             cover = f"/images/{images[0].filename}" if images else None
# # #             date_str = ""
# # #             if album.start_date:
# # #                 date_str = album.start_date.strftime("%b %Y")
# # #                 if album.end_date and album.end_date.month != album.start_date.month:
# # #                     date_str += f" – {album.end_date.strftime('%b %Y')}"
# # #             return {
# # #                 "id": album.id, "title": album.title, "type": album.type,
# # #                 "description": album.description, "date": date_str, "cover": cover,
# # #                 "start_date": album.start_date.isoformat() if album.start_date else None,
# # #                 "end_date": album.end_date.isoformat() if album.end_date else None,
# # #                 "image_count": len(images),
# # #                 "images": [{"id": img.id, "filename": img.filename, "thumbnail": f"/images/{img.filename}", "date": img.timestamp.isoformat() if img.timestamp else None} for img in images],
# # #             }
# # #         else:
# # #             albums = db.query(Album).all()
# # #             results = []
# # #             for a in albums:
# # #                 album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
# # #                 cover = f"/images/{album_images[0].filename}" if album_images else None
# # #                 date_str = ""
# # #                 if a.start_date:
# # #                     date_str = a.start_date.strftime("%b %Y")
# # #                     if a.end_date and a.end_date.month != a.start_date.month:
# # #                         date_str += f" – {a.end_date.strftime('%b %Y')}"
# # #                 results.append({
# # #                     "id": a.id, "title": a.title, "type": a.type, "description": a.description,
# # #                     "date": date_str, "cover": cover, "count": len(album_images),
# # #                     "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]],
# # #                 })
# # #             return {"results": results, "count": len(results)}
# # #     finally:
# # #         db.close()

# # # @app.post("/favorites")
# # # def add_favorite(image_id: int = Form(...)):
# # #     db = SessionLocal()
# # #     try:
# # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # #         if not img:
# # #             raise HTTPException(status_code=404)
# # #         img.is_favorite = not getattr(img, 'is_favorite', False)
# # #         db.commit()
# # #         return {"status": "success"}
# # #     finally:
# # #         db.close()

# # # @app.get("/favorites")
# # # def get_favorites():
# # #     db = SessionLocal()
# # #     try:
# # #         images = db.query(DBImage).filter(DBImage.is_favorite == True).all()
# # #         return {"count": len(images), "results": [{"id": img.id, "filename": img.filename} for img in images]}
# # #     finally:
# # #         db.close()

# # # @app.get("/duplicates")
# # # def get_duplicates():
# # #     db = SessionLocal()
# # #     try:
# # #         all_images = db.query(DBImage).all()
# # #         return {"status": "found", "duplicate_groups": [], "total_groups": 0}
# # #     finally:
# # #         db.close()

# # # @app.post("/recluster")
# # # def recluster():
# # #     db = SessionLocal()
# # #     try:
# # #         logger.info("🔄 Recluster: clearing old assignments…")
# # #         db.query(DBFace).update({"person_id": None})
# # #         db.query(Person).delete()
# # #         db.query(DBImage).update({"album_id": None})
# # #         db.query(Album).filter(Album.type == "event").delete()
# # #         db.commit()

# # #         # ── Face clustering ──────────────────────────────────────────────────
# # #         face_records = db.query(DBFace).filter(DBFace.face_embedding != None).all()
# # #         embeddings = []
# # #         valid_face_records = []
# # #         for fr in face_records:
# # #             try:
# # #                 emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
# # #                 if emb.shape[0] == 512:
# # #                     embeddings.append(emb)
# # #                     valid_face_records.append(fr)
# # #             except Exception as e:
# # #                 logger.warning(f"Bad embedding face {fr.id}: {e}")

# # #         people_count = 0
# # #         if embeddings:
# # #             logger.info(f"👥 Clustering {len(embeddings)} face embeddings…")
# # #             labels = face_engine.cluster_faces(embeddings)
# # #             person_map = {}
# # #             for i, label in enumerate(labels):
# # #                 if label == -1:
# # #                     continue
# # #                 if label not in person_map:
# # #                     p = Person(name=f"Person {label + 1}")
# # #                     db.add(p)
# # #                     db.flush()
# # #                     person_map[label] = p.id
# # #                     people_count += 1
# # #                 valid_face_records[i].person_id = person_map[label]
# # #             db.commit()
# # #             logger.info(f"✅ {people_count} people")
# # #         else:
# # #             logger.warning("⚠️  No face embeddings — run build_index.py first")

# # #         # ── Album / event detection ──────────────────────────────────────────
# # #         all_images = db.query(DBImage).all()
# # #         albums_count = 0
# # #         if all_images:
# # #             metadata = [
# # #                 {"id": img.id, "lat": img.lat or 0.0, "lon": img.lon or 0.0, "timestamp": img.timestamp}
# # #                 for img in all_images if img.timestamp
# # #             ]
# # #             if metadata:
# # #                 album_labels = clustering_engine.detect_events(metadata)
# # #                 album_map = {}
# # #                 for i, label in enumerate(album_labels):
# # #                     if label == -1:
# # #                         continue
# # #                     if label not in album_map:
# # #                         cluster_meta = [metadata[j] for j, l in enumerate(album_labels) if l == label]
# # #                         ts_list = [m["timestamp"] for m in cluster_meta if m["timestamp"]]
# # #                         start_d = min(ts_list) if ts_list else None
# # #                         end_d   = max(ts_list) if ts_list else None
# # #                         if start_d:
# # #                             if end_d and end_d.date() != start_d.date():
# # #                                 title = f"{start_d.strftime('%b %-d')} – {end_d.strftime('%b %-d, %Y')}"
# # #                             else:
# # #                                 title = start_d.strftime("%b %-d, %Y")
# # #                         else:
# # #                             title = f"Event {label + 1}"
# # #                         new_album = Album(title=title, type="event", start_date=start_d, end_date=end_d)
# # #                         db.add(new_album)
# # #                         db.flush()
# # #                         album_map[label] = new_album.id
# # #                         albums_count += 1
# # #                     img_id = metadata[i]["id"]
# # #                     db.query(DBImage).filter(DBImage.id == img_id).update({"album_id": album_map[label]})
# # #                 db.commit()
# # #                 logger.info(f"✅ {albums_count} albums")

# # #         return {"status": "done", "people": people_count, "albums": albums_count}
# # #     except Exception as e:
# # #         db.rollback()
# # #         logger.error(f"❌ Recluster failed: {e}", exc_info=True)
# # #         raise HTTPException(status_code=500, detail=str(e))
# # #     finally:
# # #         db.close()

# # # @app.post("/upload")
# # # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# # #     ext = os.path.splitext(file.filename)[1].lower()
# # #     if ext not in [".jpg", ".jpeg", ".png"]:
# # #         raise HTTPException(status_code=400)
# # #     filename = f"{uuid.uuid4()}{ext}"
# # #     file_path = os.path.join(IMAGE_DIR, filename)
# # #     db = SessionLocal()
# # #     try:
# # #         with open(file_path, "wb") as buffer:
# # #             shutil.copyfileobj(file.file, buffer)
# # #         from PIL import Image as PILImage
# # #         import json as _json
# # #         width = height = None
# # #         avg_r = avg_g = avg_b = 0.0
# # #         try:
# # #             img_pil = PILImage.open(file_path).convert("RGB")
# # #             width, height = img_pil.size
# # #             arr = np.array(img_pil)
# # #             avg_r, avg_g, avg_b = float(arr[:,:,0].mean()), float(arr[:,:,1].mean()), float(arr[:,:,2].mean())
# # #         except Exception: pass
# # #         clip_emb = None
# # #         try: clip_emb = search_engine.get_image_embedding(file_path)
# # #         except Exception: pass
# # #         ocr_text = ""
# # #         try: ocr_text = extract_text(file_path)
# # #         except Exception: pass
# # #         scene_label = ""
# # #         person_count = 0
# # #         try:
# # #             objects = detector_engine.detect_objects(file_path, threshold=0.5)
# # #             scene_label = ", ".join(objects) if objects else ""
# # #             person_count = sum(1 for o in objects if o == "person")
# # #         except Exception:
# # #             try: person_count = detector_engine.detect_persons(file_path)
# # #             except Exception: pass
# # #         img_record = DBImage(
# # #             filename=filename, original_path=file_path, timestamp=datetime.now(),
# # #             ocr_text=ocr_text, scene_label=scene_label, person_count=person_count,
# # #             width=width, height=height, avg_r=avg_r, avg_g=avg_g, avg_b=avg_b,
# # #         )
# # #         db.add(img_record)
# # #         db.flush()
# # #         if clip_emb is not None:
# # #             try:
# # #                 if search_engine.index is None:
# # #                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
# # #                 new_vec = clip_emb.reshape(1, -1).astype('float32')
# # #                 faiss.normalize_L2(new_vec)
# # #                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id]).astype('int64'))
# # #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# # #             except Exception as e: logger.warning(f"FAISS update failed: {e}")
# # #         face_count = 0
# # #         try:
# # #             faces = face_engine.detect_faces(file_path)
# # #             for face in faces:
# # #                 emb = face["embedding"].astype(np.float32)
# # #                 db.add(DBFace(image_id=img_record.id, bbox=_json.dumps(face["bbox"]), face_embedding=emb.tobytes()))
# # #                 face_count += 1
# # #         except Exception as e: logger.warning(f"Face detection failed: {e}")
# # #         db.commit()
# # #         if background_tasks is not None:
# # #             should_trigger_recluster(background_tasks)
# # #         return {"status": "success", "id": img_record.id, "filename": filename, "person_count": person_count, "face_count": face_count}
# # #     except Exception as e:
# # #         db.rollback()
# # #         if os.path.exists(file_path): os.remove(file_path)
# # #         logger.error(f"Upload failed: {e}", exc_info=True)
# # #         raise HTTPException(status_code=500)
# # #     finally:
# # #         db.close()

# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.staticfiles import StaticFiles
# # import os, uuid, shutil, numpy as np, faiss
# # from datetime import datetime
# # import logging
# # from contextlib import asynccontextmanager
# # import datetime as datetime_module

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger("main")

# # from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
# # from search_engine import search_engine, resolve_query
# # from voice_engine import voice_engine
# # from face_engine import face_engine
# # from ocr_engine import extract_text
# # from detector_engine import detector_engine
# # from duplicate_engine import duplicate_engine
# # from clustering_engine import clustering_engine

# # IMAGE_DIR = "../data/images"
# # FAISS_INDEX_PATH = "../data/index.faiss"

# # FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
# # FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
# # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # CLIP_SCORE_MIN  = float(os.environ.get("CLIP_SCORE_MIN",  0.22))
# # THRESHOLD_FLOOR = float(os.environ.get("THRESHOLD_FLOOR", 0.22))
# # ADAPTIVE_RATIO  = float(os.environ.get("ADAPTIVE_RATIO",  0.92))


# # RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # recluster_last_triggered = None

# # def should_trigger_recluster(background_tasks):
# #     global recluster_last_triggered
# #     if not RECLUSTER_ON_UPLOAD or not background_tasks:
# #         return
# #     try:
# #         counter = 0
# #         if os.path.exists(RECLUSTER_COUNTER_PATH):
# #             try:
# #                 with open(RECLUSTER_COUNTER_PATH, 'r') as f:
# #                     counter = int(f.read().strip())
# #             except:
# #                 pass
# #         counter += 1
# #         with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# #             f.write(str(counter))
# #         should_trigger = counter >= RECLUSTER_BATCH_SIZE
# #         now = datetime_module.datetime.now()
# #         if recluster_last_triggered:
# #             elapsed = (now - recluster_last_triggered).total_seconds()
# #             if elapsed >= RECLUSTER_TIMER_SECONDS:
# #                 should_trigger = True
# #         elif counter > 0:
# #             should_trigger = counter >= RECLUSTER_BATCH_SIZE
# #         if should_trigger:
# #             logger.info(f"📊 Recluster triggered")
# #             background_tasks.add_task(recluster)
# #             recluster_last_triggered = now
# #             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# #                 f.write('0')
# #     except Exception as e:
# #         logger.warning(f"Recluster check failed: {e}")

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     init_db()
# #     logger.info("🚀 Starting with SMART TWO-PASS search...")
# #     if os.path.exists(FAISS_INDEX_PATH):
# #         try:
# #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# #             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
# #         except Exception as e:
# #             logger.error(f"Index load failed: {e}")
# #             search_engine.index = None
# #     logger.info("✅ Ready!")
# #     yield

# # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)
# # app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# # if not os.path.exists(IMAGE_DIR):
# #     os.makedirs(IMAGE_DIR)
# # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # @app.get("/health")
# # def health():
# #     return {"status": "ready", "mode": "SMART_TWO_PASS", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# # @app.get("/test-db")
# # def test_db():
# #     db = SessionLocal()
# #     try:
# #         count = db.query(DBImage).count()
# #         images = db.query(DBImage).limit(1).all()
# #         return {"status": "ok", "total_images": count, "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None}
# #     except Exception as e:
# #         return {"status": "error", "message": str(e)}
# #     finally:
# #         db.close()

# # @app.post("/search")
# # def search(query: str = Form(...), top_k: int = Form(20)):
# #     """
# #     SMART TWO-PASS SEARCH:
# #     Pass 1: Find keyword-confirmed images, get minimum CLIP score
# #     Pass 2: Use that minimum as threshold for ALL images
# #     Result: ONLY relevant images shown!
# #     """
# #     if not query or not query.strip():
# #         return {"status": "error", "message": "Query empty"}

# #     processed_query = resolve_query(query)
# #     logger.info(f"🔍 Search: '{query}'")

# #     COLOR_MAP = {
# #         'red': (1.0, 0, 0), 'blue': (0, 0, 1.0), 'green': (0, 1.0, 0),
# #         'yellow': (1.0, 1.0, 0), 'orange': (1.0, 0.5, 0), 'purple': (0.5, 0, 0.5),
# #         'pink': (1.0, 0.75, 0.8), 'black': (0, 0, 0), 'white': (1, 1, 1),
# #         'gray': (0.5, 0.5, 0.5), 'brown': (0.6, 0.4, 0.2)
# #     }
# #     query_lower = query.lower()
# #     query_colors = [rgb for name, rgb in COLOR_MAP.items() if name in query_lower]

# #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# #     if query_emb is None or search_engine.index is None:
# #         return {"status": "error", "message": "No images indexed"}

# #     candidate_k = min(top_k * 8, 250)
# #     query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
# #     faiss.normalize_L2(query_emb_reshaped)
# #     distances, indices = search_engine.index.search(query_emb_reshaped, candidate_k)

# #     db = SessionLocal()
# #     try:
# #         # ===== PASS 1: Collect ALL candidates with bonuses =====
# #         all_candidates = []

# #         for dist, idx in zip(distances[0], indices[0]):
# #             if idx == -1:
# #                 continue
# #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# #             if not img:
# #                 continue

# #             raw_sim = float(dist)
# #             clip_score = max(0.0, raw_sim)
            
# #             if clip_score < CLIP_SCORE_MIN:
# #                 continue

# #             # OCR matching
# #             ocr_text = (img.ocr_text or "").lower()
# #             query_words = processed_query.lower().split()
# #             significant_words = [w for w in query_words if len(w) > 2]
            
# #             ocr_bonus = 0.0
# #             if significant_words and ocr_text:
# #                 matches = sum(1 for w in significant_words if w in ocr_text)
# #                 if matches > 0:
# #                     ocr_bonus = min(matches / len(significant_words), 1.0)

# #             # Tag matching — exact word only (substring caused "man"→"woman", "dog"→"hotdog")
# #             tag_bonus = 0.0
# #             if img.scene_label:
# #                 tag_words = set()
# #                 for tag in img.scene_label.split(","):
# #                     for word in tag.strip().lower().split():
# #                         tag_words.add(word)
# #                 query_objects = [w for w in query_words if len(w) > 2]
# #                 if any(obj in tag_words for obj in query_objects):
# #                     tag_bonus = 1.0

# #             # Color matching
# #             color_bonus = 0.0
# #             if query_colors and getattr(img, 'avg_r', None) is not None:
# #                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# #                 for qc in query_colors:
# #                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# #                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
# #                     color_bonus = max(color_bonus, score)

# #             has_keyword_match = ocr_bonus > 0 or tag_bonus > 0
            
# #             if has_keyword_match:
# #                 final_score = (
# #                     (0.40 * clip_score) +
# #                     (0.35 * ocr_bonus) +
# #                     (0.15 * color_bonus) +
# #                     (0.10 * tag_bonus)
# #                 )
# #             else:
# #                 final_score = clip_score

# #             all_candidates.append({
# #                 'img': img,
# #                 'clip_score': clip_score,
# #                 'final_score': final_score,
# #                 'ocr_bonus': ocr_bonus,
# #                 'tag_bonus': tag_bonus,
# #                 'color_bonus': color_bonus,
# #                 'has_keyword_match': has_keyword_match
# #             })

# #         # ===== CALCULATE THRESHOLD =====
# #         keyword_confirmed_clips = [c['clip_score'] for c in all_candidates if c['has_keyword_match']]

# #         if keyword_confirmed_clips:
# #             threshold = max(min(keyword_confirmed_clips), THRESHOLD_FLOOR)
# #             logger.info(f"🎯 Threshold from {len(keyword_confirmed_clips)} confirmed: {threshold:.4f}")
# #         else:
# #             all_clips = [c['clip_score'] for c in all_candidates]
# #             if all_clips:
# #                 best_clip = max(all_clips)
# #                 threshold = max(best_clip * ADAPTIVE_RATIO, THRESHOLD_FLOOR)
# #                 logger.info(f"⚠️  Adaptive threshold: {threshold:.4f}")
# #             else:
# #                 return {"status": "not_found", "message": f"No images for '{query}'"}

# #         # ===== PASS 2: Filter by threshold =====
# #         results = []

# #         for candidate in all_candidates:
# #             img = candidate['img']
# #             clip_score = candidate['clip_score']
# #             final_score = candidate['final_score']
# #             has_keyword_match = candidate['has_keyword_match']

# #             # Every image must pass the threshold — keyword match only affects score weight.
# #             if clip_score < threshold:
# #                 logger.debug(f"❌ Skip {img.filename}: CLIP {clip_score:.4f} < {threshold:.4f}")
# #                 continue
            
# #             logger.debug(f"✅ {img.filename}: CLIP={clip_score:.4f}, FINAL={final_score:.4f}")
            
# #             results.append({
# #                 "id": img.id,
# #                 "filename": img.filename,
# #                 "score": round(final_score * 100, 2),
# #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# #                 "person_count": img.person_count or 0
# #             })

# #         results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
# #         if not results:
# #             return {"status": "not_found", "message": f"No match for '{query}'"}

# #         logger.info(f"✅ Found {len(results)} results")
# #         return {"status": "found", "query": query, "count": len(results), "threshold": round(threshold * 100, 1), "results": results}
# #     finally:
# #         db.close()

# # @app.post("/search/voice")
# # def voice_search(duration: int = Form(5)):
# #     try:
# #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# #         if not transcribed:
# #             return {"status": "error", "message": "No audio"}
# #         return search(query=transcribed, top_k=20)
# #     except Exception as e:
# #         return {"status": "error", "message": str(e)}

# # @app.get("/timeline")
# # def get_timeline():
# #     db = SessionLocal()
# #     try:
# #         images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
# #         results = [{"id": img.id, "filename": img.filename, "date": img.timestamp.isoformat() if img.timestamp else None, "thumbnail": f"/images/{img.filename}"} for img in images]
# #         return {"count": len(results), "results": results}
# #     finally:
# #         db.close()

# # @app.get("/faces")
# # def get_faces(person_id: int = Query(None)):
# #     db = SessionLocal()
# #     try:
# #         if person_id:
# #             person = db.query(Person).filter(Person.id == person_id).first()
# #             if not person:
# #                 raise HTTPException(status_code=404)
# #             faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# #             images = []
# #             cover = None
# #             for f in faces:
# #                 img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# #                 if img:
# #                     if cover is None:
# #                         cover = img.filename
# #                     images.append({
# #                         "id": img.id,
# #                         "filename": img.filename,
# #                         "thumbnail": img.filename,
# #                         "date": img.timestamp.isoformat() if img.timestamp else None,
# #                     })
# #             return {"id": person.id, "name": person.name, "face_count": len(faces), "cover": cover, "images": images}
# #         else:
# #             people = db.query(Person).all()
# #             results = []
# #             for p in people:
# #                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# #                 if not faces:
# #                     continue
# #                 cover = None
# #                 for f in faces:
# #                     img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# #                     if img and img.filename:
# #                         cover = img.filename
# #                         break
# #                 results.append({"id": p.id, "name": p.name, "count": len(faces), "cover": cover})
# #             return {"results": results, "count": len(results)}
# #     finally:
# #         db.close()

# # @app.post("/people/{person_id}")
# # def update_person(person_id: int, name: str = Form(...)):
# #     db = SessionLocal()
# #     try:
# #         person = db.query(Person).filter(Person.id == person_id).first()
# #         if not person:
# #             raise HTTPException(status_code=404)
# #         person.name = name
# #         db.commit()
# #         return {"status": "success", "id": person.id, "name": person.name}
# #     finally:
# #         db.close()

# # @app.get("/people/{person_id}/celebcheck")
# # def check_celebrity_match(person_id: int):
# #     db = SessionLocal()
# #     try:
# #         person = db.query(Person).filter(Person.id == person_id).first()
# #         if not person:
# #             return {"status": "no_match"}
# #         return {"status": "no_match"}
# #     finally:
# #         db.close()

# # @app.get("/albums")
# # def get_albums(album_id: int = Query(None)):
# #     db = SessionLocal()
# #     try:
# #         if album_id:
# #             album = db.query(Album).filter(Album.id == album_id).first()
# #             if not album:
# #                 raise HTTPException(status_code=404)
# #             images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
# #             cover = images[0].filename if images else None
# #             date_str = ""
# #             if album.start_date:
# #                 date_str = album.start_date.strftime("%b %Y")
# #                 if album.end_date and album.end_date.month != album.start_date.month:
# #                     date_str += f" – {album.end_date.strftime('%b %Y')}"
# #             return {
# #                 "id": album.id, "title": album.title, "type": album.type,
# #                 "description": album.description, "date": date_str, "cover": cover,
# #                 "start_date": album.start_date.isoformat() if album.start_date else None,
# #                 "end_date": album.end_date.isoformat() if album.end_date else None,
# #                 "image_count": len(images),
# #                 "images": [{"id": img.id, "filename": img.filename, "thumbnail": img.filename, "date": img.timestamp.isoformat() if img.timestamp else None} for img in images],
# #             }
# #         else:
# #             albums = db.query(Album).all()
# #             results = []
# #             for a in albums:
# #                 album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
# #                 cover = album_images[0].filename if album_images else None
# #                 date_str = ""
# #                 if a.start_date:
# #                     date_str = a.start_date.strftime("%b %Y")
# #                     if a.end_date and a.end_date.month != a.start_date.month:
# #                         date_str += f" – {a.end_date.strftime('%b %Y')}"
# #                 results.append({
# #                     "id": a.id, "title": a.title, "type": a.type, "description": a.description,
# #                     "date": date_str, "cover": cover, "count": len(album_images),
# #                     "thumbnails": [img.filename for img in album_images[:4]],
# #                 })
# #             return {"results": results, "count": len(results)}
# #     finally:
# #         db.close()

# # @app.post("/favorites")
# # def add_favorite(image_id: int = Form(...)):
# #     db = SessionLocal()
# #     try:
# #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# #         if not img:
# #             raise HTTPException(status_code=404)
# #         img.is_favorite = not getattr(img, 'is_favorite', False)
# #         db.commit()
# #         return {"status": "success"}
# #     finally:
# #         db.close()

# # @app.get("/favorites")
# # def get_favorites():
# #     db = SessionLocal()
# #     try:
# #         images = db.query(DBImage).filter(DBImage.is_favorite == True).all()
# #         return {"count": len(images), "results": [{"id": img.id, "filename": img.filename} for img in images]}
# #     finally:
# #         db.close()

# # @app.get("/duplicates")
# # def get_duplicates():
# #     db = SessionLocal()
# #     try:
# #         all_images = db.query(DBImage).all()
# #         return {"status": "found", "duplicate_groups": [], "total_groups": 0}
# #     finally:
# #         db.close()

# # @app.post("/recluster")
# # def recluster():
# #     db = SessionLocal()
# #     try:
# #         logger.info("🔄 Recluster: clearing old assignments…")
# #         db.query(DBFace).update({"person_id": None})
# #         db.query(Person).delete()
# #         db.query(DBImage).update({"album_id": None})
# #         db.query(Album).filter(Album.type == "event").delete()
# #         db.commit()

# #         # ── Face clustering ──────────────────────────────────────────────────
# #         face_records = db.query(DBFace).filter(DBFace.face_embedding != None).all()
# #         embeddings = []
# #         valid_face_records = []
# #         for fr in face_records:
# #             try:
# #                 emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
# #                 if emb.shape[0] == 512:
# #                     embeddings.append(emb)
# #                     valid_face_records.append(fr)
# #             except Exception as e:
# #                 logger.warning(f"Bad embedding face {fr.id}: {e}")

# #         people_count = 0
# #         if embeddings:
# #             logger.info(f"👥 Clustering {len(embeddings)} face embeddings…")
# #             labels = face_engine.cluster_faces(embeddings)
# #             person_map = {}
# #             for i, label in enumerate(labels):
# #                 if label == -1:
# #                     continue
# #                 if label not in person_map:
# #                     p = Person(name=f"Person {label + 1}")
# #                     db.add(p)
# #                     db.flush()
# #                     person_map[label] = p.id
# #                     people_count += 1
# #                 valid_face_records[i].person_id = person_map[label]
# #             db.commit()
# #             logger.info(f"✅ {people_count} people")
# #         else:
# #             logger.warning("⚠️  No face embeddings — run build_index.py first")

# #         # ── Album / event detection ──────────────────────────────────────────
# #         all_images = db.query(DBImage).all()
# #         albums_count = 0
# #         if all_images:
# #             metadata = [
# #                 {"id": img.id, "lat": img.lat or 0.0, "lon": img.lon or 0.0, "timestamp": img.timestamp}
# #                 for img in all_images if img.timestamp
# #             ]
# #             if metadata:
# #                 album_labels = clustering_engine.detect_events(metadata)
# #                 album_map = {}
# #                 for i, label in enumerate(album_labels):
# #                     if label == -1:
# #                         continue
# #                     if label not in album_map:
# #                         cluster_meta = [metadata[j] for j, l in enumerate(album_labels) if l == label]
# #                         ts_list = [m["timestamp"] for m in cluster_meta if m["timestamp"]]
# #                         start_d = min(ts_list) if ts_list else None
# #                         end_d   = max(ts_list) if ts_list else None
# #                         if start_d:
# #                             if end_d and end_d.date() != start_d.date():
# #                                 title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%b %d, %Y')}"
# #                             else:
# #                                 title = start_d.strftime("%b %d, %Y")
# #                         else:
# #                             title = f"Event {label + 1}"
# #                         new_album = Album(title=title, type="event", start_date=start_d, end_date=end_d)
# #                         db.add(new_album)
# #                         db.flush()
# #                         album_map[label] = new_album.id
# #                         albums_count += 1
# #                     img_id = metadata[i]["id"]
# #                     db.query(DBImage).filter(DBImage.id == img_id).update({"album_id": album_map[label]})
# #                 db.commit()
# #                 logger.info(f"✅ {albums_count} albums")

# #         return {"status": "done", "people": people_count, "albums": albums_count}
# #     except Exception as e:
# #         db.rollback()
# #         logger.error(f"❌ Recluster failed: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=str(e))
# #     finally:
# #         db.close()

# # @app.post("/upload")
# # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# #     ext = os.path.splitext(file.filename)[1].lower()
# #     if ext not in [".jpg", ".jpeg", ".png"]:
# #         raise HTTPException(status_code=400)
# #     filename = f"{uuid.uuid4()}{ext}"
# #     file_path = os.path.join(IMAGE_DIR, filename)
# #     db = SessionLocal()
# #     try:
# #         with open(file_path, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)
# #         from PIL import Image as PILImage
# #         import json as _json
# #         width = height = None
# #         avg_r = avg_g = avg_b = 0.0
# #         try:
# #             img_pil = PILImage.open(file_path).convert("RGB")
# #             width, height = img_pil.size
# #             arr = np.array(img_pil)
# #             avg_r, avg_g, avg_b = float(arr[:,:,0].mean()), float(arr[:,:,1].mean()), float(arr[:,:,2].mean())
# #         except Exception: pass
# #         clip_emb = None
# #         try: clip_emb = search_engine.get_image_embedding(file_path)
# #         except Exception: pass
# #         ocr_text = ""
# #         try: ocr_text = extract_text(file_path)
# #         except Exception: pass
# #         scene_label = ""
# #         person_count = 0
# #         try:
# #             objects = detector_engine.detect_objects(file_path, threshold=0.5)
# #             scene_label = ", ".join(objects) if objects else ""
# #             person_count = sum(1 for o in objects if o == "person")
# #         except Exception:
# #             try: person_count = detector_engine.detect_persons(file_path)
# #             except Exception: pass
# #         img_record = DBImage(
# #             filename=filename, original_path=file_path, timestamp=datetime.now(),
# #             ocr_text=ocr_text, scene_label=scene_label, person_count=person_count,
# #             width=width, height=height, avg_r=avg_r, avg_g=avg_g, avg_b=avg_b,
# #         )
# #         db.add(img_record)
# #         db.flush()
# #         if clip_emb is not None:
# #             try:
# #                 if search_engine.index is None:
# #                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
# #                 new_vec = clip_emb.reshape(1, -1).astype('float32')
# #                 faiss.normalize_L2(new_vec)
# #                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id]).astype('int64'))
# #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# #             except Exception as e: logger.warning(f"FAISS update failed: {e}")
# #         face_count = 0
# #         try:
# #             faces = face_engine.detect_faces(file_path)
# #             for face in faces:
# #                 emb = face["embedding"].astype(np.float32)
# #                 db.add(DBFace(image_id=img_record.id, bbox=_json.dumps(face["bbox"]), face_embedding=emb.tobytes()))
# #                 face_count += 1
# #         except Exception as e: logger.warning(f"Face detection failed: {e}")
# #         db.commit()
# #         if background_tasks is not None:
# #             should_trigger_recluster(background_tasks)
# #         return {"status": "success", "id": img_record.id, "filename": filename, "person_count": person_count, "face_count": face_count}
# #     except Exception as e:
# #         db.rollback()
# #         if os.path.exists(file_path): os.remove(file_path)
# #         logger.error(f"Upload failed: {e}", exc_info=True)
# #         raise HTTPException(status_code=500)
# #     finally:
# #         db.close()

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# import os, uuid, shutil, numpy as np, faiss
# from datetime import datetime
# import logging
# from contextlib import asynccontextmanager
# import datetime as datetime_module

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("main")

# from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
# from search_engine import search_engine, resolve_query
# from voice_engine import voice_engine
# from face_engine import face_engine
# from ocr_engine import extract_text
# from detector_engine import detector_engine
# from duplicate_engine import duplicate_engine
# from clustering_engine import clustering_engine

# IMAGE_DIR = "../data/images"
# FAISS_INDEX_PATH = "../data/index.faiss"

# FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
# FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
# FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# CLIP_SCORE_MIN  = float(os.environ.get("CLIP_SCORE_MIN",  0.22))
# THRESHOLD_FLOOR = float(os.environ.get("THRESHOLD_FLOOR", 0.22))
# ADAPTIVE_RATIO  = float(os.environ.get("ADAPTIVE_RATIO",  0.92))


# RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# recluster_last_triggered = None

# def should_trigger_recluster(background_tasks):
#     global recluster_last_triggered
#     if not RECLUSTER_ON_UPLOAD or not background_tasks:
#         return
#     try:
#         counter = 0
#         if os.path.exists(RECLUSTER_COUNTER_PATH):
#             try:
#                 with open(RECLUSTER_COUNTER_PATH, 'r') as f:
#                     counter = int(f.read().strip())
#             except:
#                 pass
#         counter += 1
#         with open(RECLUSTER_COUNTER_PATH, 'w') as f:
#             f.write(str(counter))
#         should_trigger = counter >= RECLUSTER_BATCH_SIZE
#         now = datetime_module.datetime.now()
#         if recluster_last_triggered:
#             elapsed = (now - recluster_last_triggered).total_seconds()
#             if elapsed >= RECLUSTER_TIMER_SECONDS:
#                 should_trigger = True
#         elif counter > 0:
#             should_trigger = counter >= RECLUSTER_BATCH_SIZE
#         if should_trigger:
#             logger.info(f"📊 Recluster triggered")
#             background_tasks.add_task(recluster)
#             recluster_last_triggered = now
#             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
#                 f.write('0')
#     except Exception as e:
#         logger.warning(f"Recluster check failed: {e}")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     init_db()
#     logger.info("🚀 Starting with SMART TWO-PASS search...")
#     if os.path.exists(FAISS_INDEX_PATH):
#         try:
#             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
#             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
#         except Exception as e:
#             logger.error(f"Index load failed: {e}")
#             search_engine.index = None
#     logger.info("✅ Ready!")
#     yield

# app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# if not os.path.exists(IMAGE_DIR):
#     os.makedirs(IMAGE_DIR)
# app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# @app.get("/health")
# def health():
#     return {"status": "ready", "mode": "SMART_TWO_PASS", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# @app.get("/test-db")
# def test_db():
#     db = SessionLocal()
#     try:
#         count = db.query(DBImage).count()
#         images = db.query(DBImage).limit(1).all()
#         return {"status": "ok", "total_images": count, "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
#     finally:
#         db.close()

# @app.post("/search")
# def search(query: str = Form(...), top_k: int = Form(20)):
#     """
#     Simple, reliable CLIP-based search.
#     1. Get top candidates from FAISS (cosine similarity, already normalized).
#     2. Apply a hard CLIP_SCORE_MIN floor — nothing below this ever appears.
#     3. Add small OCR / tag bonuses to final score for ordering only.
#     4. Return top_k results sorted by final score.

#     The previous "two-pass adaptive threshold" caused unrelated images to
#     appear because with a small gallery (47 images) almost all images score
#     above the adaptive floor and get returned.
#     """
#     if not query or not query.strip():
#         return {"status": "error", "message": "Query empty"}

#     processed_query = resolve_query(query)
#     logger.info(f"🔍 Search: '{query}' → '{processed_query}'")

#     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
#     if query_emb is None or search_engine.index is None:
#         return {"status": "error", "message": "No images indexed. Run build_index.py first."}

#     # Fetch more candidates than needed so we have headroom after filtering
#     candidate_k = min(top_k * 5, search_engine.index.ntotal)
#     q = query_emb.reshape(1, -1).astype('float32')
#     faiss.normalize_L2(q)
#     distances, indices = search_engine.index.search(q, candidate_k)

#     db = SessionLocal()
#     try:
#         results = []
#         query_words = processed_query.lower().split()
#         sig_words = [w for w in query_words if len(w) > 2]

#         for dist, idx in zip(distances[0], indices[0]):
#             if idx == -1:
#                 continue

#             clip_score = float(dist)

#             # Hard floor — unrelated images are always below this
#             if clip_score < CLIP_SCORE_MIN:
#                 break  # FAISS returns results sorted by score; once below floor, all rest are too

#             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
#             if not img:
#                 continue

#             # OCR bonus — small boost when query words appear in image text
#             ocr_bonus = 0.0
#             if sig_words and img.ocr_text:
#                 ocr_text = img.ocr_text.lower()
#                 matches = sum(1 for w in sig_words if w in ocr_text)
#                 ocr_bonus = min(matches / len(sig_words), 1.0)

#             # Tag bonus — exact word match only (no substrings)
#             tag_bonus = 0.0
#             if img.scene_label:
#                 tag_words = set()
#                 for tag in img.scene_label.split(","):
#                     for word in tag.strip().lower().split():
#                         tag_words.add(word)
#                 if any(w in tag_words for w in sig_words):
#                     tag_bonus = 1.0

#             # Final score: CLIP dominates (80%), bonuses only break ties
#             final_score = (0.80 * clip_score) + (0.12 * ocr_bonus) + (0.08 * tag_bonus)

#             logger.debug(f"✅ {img.filename}: CLIP={clip_score:.3f} OCR={ocr_bonus:.2f} TAG={tag_bonus:.2f} → {final_score:.3f}")

#             results.append({
#                 "id": img.id,
#                 "filename": img.filename,
#                 "score": round(final_score * 100, 2),
#                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
#                 "person_count": img.person_count or 0,
#             })

#         results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]

#         if not results:
#             return {"status": "not_found", "message": f"No images matched '{query}'. Try different keywords."}

#         logger.info(f"✅ {len(results)} results for '{query}' (threshold={CLIP_SCORE_MIN})")
#         return {"status": "found", "query": query, "count": len(results), "results": results}
#     finally:
#         db.close()

# @app.post("/search/voice")
# def voice_search(duration: int = Form(5)):
#     try:
#         transcribed = voice_engine.listen_and_transcribe(duration=duration)
#         if not transcribed:
#             return {"status": "error", "message": "No audio"}
#         return search(query=transcribed, top_k=20)
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# @app.get("/timeline")
# def get_timeline():
#     db = SessionLocal()
#     try:
#         images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
#         results = [{"id": img.id, "filename": img.filename, "date": img.timestamp.isoformat() if img.timestamp else None, "thumbnail": f"/images/{img.filename}"} for img in images]
#         return {"count": len(results), "results": results}
#     finally:
#         db.close()

# @app.get("/faces")
# def get_faces(person_id: int = Query(None)):
#     db = SessionLocal()
#     try:
#         if person_id:
#             person = db.query(Person).filter(Person.id == person_id).first()
#             if not person:
#                 raise HTTPException(status_code=404)
#             faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
#             images = []
#             cover = None
#             for f in faces:
#                 img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
#                 if img:
#                     if cover is None:
#                         cover = f"/images/{img.filename}"
#                     images.append({
#                         "id": img.id,
#                         "filename": img.filename,
#                         "thumbnail": f"/images/{img.filename}",
#                         "date": img.timestamp.isoformat() if img.timestamp else None,
#                     })
#             return {"id": person.id, "name": person.name, "face_count": len(faces), "cover": cover, "images": images}
#         else:
#             people = db.query(Person).all()
#             results = []
#             for p in people:
#                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
#                 if not faces:
#                     continue
#                 cover = None
#                 for f in faces:
#                     img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
#                     if img and img.filename:
#                         cover = f"/images/{img.filename}"
#                         break
#                 results.append({"id": p.id, "name": p.name, "count": len(faces), "cover": cover})
#             return {"results": results, "count": len(results)}
#     finally:
#         db.close()

# @app.get("/people/{person_id}")
# def get_person(person_id: int):
#     """Get a single person with all their images. Frontend calls this when clicking a person card."""
#     db = SessionLocal()
#     try:
#         person = db.query(Person).filter(Person.id == person_id).first()
#         if not person:
#             raise HTTPException(status_code=404)
#         faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
#         images = []
#         cover = None
#         for f in faces:
#             img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
#             if img:
#                 if cover is None:
#                     cover = f"/images/{img.filename}"
#                 images.append({
#                     "id": img.id,
#                     "filename": img.filename,
#                     "thumbnail": f"/images/{img.filename}",
#                     "date": img.timestamp.isoformat() if img.timestamp else None,
#                 })
#         return {"id": person.id, "name": person.name, "face_count": len(faces), "cover": cover, "images": images}
#     finally:
#         db.close()

# @app.post("/people/{person_id}")
# def update_person(person_id: int, name: str = Form(...)):
#     db = SessionLocal()
#     try:
#         person = db.query(Person).filter(Person.id == person_id).first()
#         if not person:
#             raise HTTPException(status_code=404)
#         person.name = name
#         db.commit()
#         return {"status": "success", "id": person.id, "name": person.name}
#     finally:
#         db.close()

# @app.get("/people/{person_id}/celebcheck")
# def check_celebrity_match(person_id: int):
#     db = SessionLocal()
#     try:
#         person = db.query(Person).filter(Person.id == person_id).first()
#         if not person:
#             return {"status": "no_match"}
#         return {"status": "no_match"}
#     finally:
#         db.close()

# @app.get("/albums/{album_id}")
# def get_album_by_id(album_id: int):
#     """Get a single album by path param — frontend navigates to /albums/1 etc."""
#     db = SessionLocal()
#     try:
#         album = db.query(Album).filter(Album.id == album_id).first()
#         if not album:
#             raise HTTPException(status_code=404)
#         images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
#         cover = f"/images/{images[0].filename}" if images else None
#         date_str = ""
#         if album.start_date:
#             date_str = album.start_date.strftime("%b %Y")
#             if album.end_date and album.end_date.month != album.start_date.month:
#                 date_str += f" – {album.end_date.strftime('%b %Y')}"
#         return {
#             "id": album.id, "title": album.title, "type": album.type,
#             "description": album.description, "date": date_str, "cover": cover,
#             "start_date": album.start_date.isoformat() if album.start_date else None,
#             "end_date": album.end_date.isoformat() if album.end_date else None,
#             "image_count": len(images),
#             "images": [{"id": img.id, "filename": img.filename, "thumbnail": f"/images/{img.filename}", "date": img.timestamp.isoformat() if img.timestamp else None} for img in images],
#         }
#     finally:
#         db.close()

# @app.get("/albums")
# def get_albums(album_id: int = Query(None)):
#     db = SessionLocal()
#     try:
#         if album_id:
#             return get_album_by_id(album_id)
#         else:
#             albums = db.query(Album).all()
#             results = []
#             for a in albums:
#                 album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
#                 cover = f"/images/{album_images[0].filename}" if album_images else None
#                 date_str = ""
#                 if a.start_date:
#                     date_str = a.start_date.strftime("%b %Y")
#                     if a.end_date and a.end_date.month != a.start_date.month:
#                         date_str += f" – {a.end_date.strftime('%b %Y')}"
#                 results.append({
#                     "id": a.id, "title": a.title, "type": a.type, "description": a.description,
#                     "date": date_str, "cover": cover, "count": len(album_images),
#                     "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]],
#                 })
#             return {"results": results, "count": len(results)}
#     finally:
#         db.close()

# @app.post("/favorites")
# def add_favorite(image_id: int = Form(...)):
#     db = SessionLocal()
#     try:
#         img = db.query(DBImage).filter(DBImage.id == image_id).first()
#         if not img:
#             raise HTTPException(status_code=404)
#         img.is_favorite = not getattr(img, 'is_favorite', False)
#         db.commit()
#         return {"status": "success"}
#     finally:
#         db.close()

# @app.get("/favorites")
# def get_favorites():
#     db = SessionLocal()
#     try:
#         images = db.query(DBImage).filter(DBImage.is_favorite == True).all()
#         return {"count": len(images), "results": [{"id": img.id, "filename": img.filename} for img in images]}
#     finally:
#         db.close()

# @app.get("/duplicates")
# def get_duplicates():
#     db = SessionLocal()
#     try:
#         all_images = db.query(DBImage).all()
#         return {"status": "found", "duplicate_groups": [], "total_groups": 0}
#     finally:
#         db.close()

# @app.post("/recluster")
# def recluster():
#     db = SessionLocal()
#     try:
#         logger.info("🔄 Recluster: clearing old assignments…")
#         db.query(DBFace).update({"person_id": None})
#         db.query(Person).delete()
#         db.query(DBImage).update({"album_id": None})
#         db.query(Album).filter(Album.type == "event").delete()
#         db.commit()

#         # ── Face clustering ──────────────────────────────────────────────────
#         face_records = db.query(DBFace).filter(DBFace.face_embedding != None).all()
#         embeddings = []
#         valid_face_records = []
#         for fr in face_records:
#             try:
#                 emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
#                 if emb.shape[0] == 512:
#                     embeddings.append(emb)
#                     valid_face_records.append(fr)
#             except Exception as e:
#                 logger.warning(f"Bad embedding face {fr.id}: {e}")

#         people_count = 0
#         if embeddings:
#             logger.info(f"👥 Clustering {len(embeddings)} face embeddings…")
#             labels = face_engine.cluster_faces(embeddings)
#             person_map = {}
#             for i, label in enumerate(labels):
#                 if label == -1:
#                     continue
#                 if label not in person_map:
#                     p = Person(name=f"Person {label + 1}")
#                     db.add(p)
#                     db.flush()
#                     person_map[label] = p.id
#                     people_count += 1
#                 valid_face_records[i].person_id = person_map[label]
#             db.commit()
#             logger.info(f"✅ {people_count} people")
#         else:
#             logger.warning("⚠️  No face embeddings — run build_index.py first")

#         # ── Album / event detection ──────────────────────────────────────────
#         all_images = db.query(DBImage).all()
#         albums_count = 0
#         if all_images:
#             metadata = [
#                 {"id": img.id, "lat": img.lat or 0.0, "lon": img.lon or 0.0, "timestamp": img.timestamp}
#                 for img in all_images if img.timestamp
#             ]
#             if metadata:
#                 album_labels = clustering_engine.detect_events(metadata)
#                 album_map = {}
#                 for i, label in enumerate(album_labels):
#                     if label == -1:
#                         continue
#                     if label not in album_map:
#                         cluster_meta = [metadata[j] for j, l in enumerate(album_labels) if l == label]
#                         ts_list = [m["timestamp"] for m in cluster_meta if m["timestamp"]]
#                         start_d = min(ts_list) if ts_list else None
#                         end_d   = max(ts_list) if ts_list else None
#                         if start_d:
#                             if end_d and end_d.date() != start_d.date():
#                                 title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%b %d, %Y')}"
#                             else:
#                                 title = start_d.strftime("%b %d, %Y")
#                         else:
#                             title = f"Event {label + 1}"
#                         new_album = Album(title=title, type="event", start_date=start_d, end_date=end_d)
#                         db.add(new_album)
#                         db.flush()
#                         album_map[label] = new_album.id
#                         albums_count += 1
#                     img_id = metadata[i]["id"]
#                     db.query(DBImage).filter(DBImage.id == img_id).update({"album_id": album_map[label]})
#                 db.commit()
#                 logger.info(f"✅ {albums_count} albums")

#         return {"status": "done", "people": people_count, "albums": albums_count}
#     except Exception as e:
#         db.rollback()
#         logger.error(f"❌ Recluster failed: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         db.close()

# @app.post("/upload")
# async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
#     ext = os.path.splitext(file.filename)[1].lower()
#     if ext not in [".jpg", ".jpeg", ".png"]:
#         raise HTTPException(status_code=400)
#     filename = f"{uuid.uuid4()}{ext}"
#     file_path = os.path.join(IMAGE_DIR, filename)
#     db = SessionLocal()
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         from PIL import Image as PILImage
#         import json as _json
#         width = height = None
#         avg_r = avg_g = avg_b = 0.0
#         try:
#             img_pil = PILImage.open(file_path).convert("RGB")
#             width, height = img_pil.size
#             arr = np.array(img_pil)
#             avg_r, avg_g, avg_b = float(arr[:,:,0].mean()), float(arr[:,:,1].mean()), float(arr[:,:,2].mean())
#         except Exception: pass
#         clip_emb = None
#         try: clip_emb = search_engine.get_image_embedding(file_path)
#         except Exception: pass
#         ocr_text = ""
#         try: ocr_text = extract_text(file_path)
#         except Exception: pass
#         scene_label = ""
#         person_count = 0
#         try:
#             objects = detector_engine.detect_objects(file_path, threshold=0.5)
#             scene_label = ", ".join(objects) if objects else ""
#             person_count = sum(1 for o in objects if o == "person")
#         except Exception:
#             try: person_count = detector_engine.detect_persons(file_path)
#             except Exception: pass
#         img_record = DBImage(
#             filename=filename, original_path=file_path, timestamp=datetime.now(),
#             ocr_text=ocr_text, scene_label=scene_label, person_count=person_count,
#             width=width, height=height, avg_r=avg_r, avg_g=avg_g, avg_b=avg_b,
#         )
#         db.add(img_record)
#         db.flush()
#         if clip_emb is not None:
#             try:
#                 if search_engine.index is None:
#                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
#                 new_vec = clip_emb.reshape(1, -1).astype('float32')
#                 faiss.normalize_L2(new_vec)
#                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id]).astype('int64'))
#                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
#             except Exception as e: logger.warning(f"FAISS update failed: {e}")
#         face_count = 0
#         try:
#             faces = face_engine.detect_faces(file_path)
#             for face in faces:
#                 emb = face["embedding"].astype(np.float32)
#                 db.add(DBFace(image_id=img_record.id, bbox=_json.dumps(face["bbox"]), face_embedding=emb.tobytes()))
#                 face_count += 1
#         except Exception as e: logger.warning(f"Face detection failed: {e}")
#         db.commit()
#         if background_tasks is not None:
#             should_trigger_recluster(background_tasks)
#         return {"status": "success", "id": img_record.id, "filename": filename, "person_count": person_count, "face_count": face_count}
#     except Exception as e:
#         db.rollback()
#         if os.path.exists(file_path): os.remove(file_path)
#         logger.error(f"Upload failed: {e}", exc_info=True)
#         raise HTTPException(status_code=500)
#     finally:
#         db.close()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, uuid, shutil, numpy as np, faiss
from datetime import datetime
import logging
from contextlib import asynccontextmanager
import datetime as datetime_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
from search_engine import search_engine, resolve_query
from voice_engine import voice_engine
from face_engine import face_engine
from ocr_engine import extract_text
from detector_engine import detector_engine
from duplicate_engine import duplicate_engine
from clustering_engine import clustering_engine

IMAGE_DIR = "../data/images"
FAISS_INDEX_PATH = "../data/index.faiss"

FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

CLIP_SCORE_MIN  = float(os.environ.get("CLIP_SCORE_MIN",  0.22))
THRESHOLD_FLOOR = float(os.environ.get("THRESHOLD_FLOOR", 0.22))
ADAPTIVE_RATIO  = float(os.environ.get("ADAPTIVE_RATIO",  0.92))


RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
recluster_last_triggered = None

def should_trigger_recluster(background_tasks):
    global recluster_last_triggered
    if not RECLUSTER_ON_UPLOAD or not background_tasks:
        return
    try:
        counter = 0
        if os.path.exists(RECLUSTER_COUNTER_PATH):
            try:
                with open(RECLUSTER_COUNTER_PATH, 'r') as f:
                    counter = int(f.read().strip())
            except:
                pass
        counter += 1
        with open(RECLUSTER_COUNTER_PATH, 'w') as f:
            f.write(str(counter))
        should_trigger = counter >= RECLUSTER_BATCH_SIZE
        now = datetime_module.datetime.now()
        if recluster_last_triggered:
            elapsed = (now - recluster_last_triggered).total_seconds()
            if elapsed >= RECLUSTER_TIMER_SECONDS:
                should_trigger = True
        elif counter > 0:
            should_trigger = counter >= RECLUSTER_BATCH_SIZE
        if should_trigger:
            logger.info(f"📊 Recluster triggered")
            background_tasks.add_task(recluster)
            recluster_last_triggered = now
            with open(RECLUSTER_COUNTER_PATH, 'w') as f:
                f.write('0')
    except Exception as e:
        logger.warning(f"Recluster check failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("🚀 Starting with SMART TWO-PASS search...")
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
        except Exception as e:
            logger.error(f"Index load failed: {e}")
            search_engine.index = None
    logger.info("✅ Ready!")
    yield

app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

@app.get("/health")
def health():
    return {"status": "ready", "mode": "SMART_TWO_PASS", "image_index": search_engine.index.ntotal if search_engine.index else 0}

@app.get("/test-db")
def test_db():
    db = SessionLocal()
    try:
        count = db.query(DBImage).count()
        images = db.query(DBImage).limit(1).all()
        return {"status": "ok", "total_images": count, "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@app.post("/search")
def search(query: str = Form(...), top_k: int = Form(20)):
    """
    Gap-detection search.

    Problem with a fixed threshold: "dog" needs ≥0.29, "cat" ≥0.26, "horse" ≥0.32.
    No single number works for all queries.

    Solution: score all candidates above a generous floor (0.20), then find the
    biggest SCORE DROP between consecutive results (sorted high→low). Everything
    above that drop is relevant; everything below is noise. The gap naturally
    varies per query.
    """
    if not query or not query.strip():
        return {"status": "error", "message": "Query empty"}

    processed_query = resolve_query(query)
    logger.info(f"🔍 Search: '{query}' → '{processed_query}'")

    # extract color words from query for color bonus
    COLOR_MAP = {
        'red': (1.0,0,0), 'blue': (0,0,1.0), 'green': (0,1.0,0),
        'yellow': (1.0,1.0,0), 'orange': (1.0,0.5,0), 'purple': (0.5,0,0.5),
        'pink': (1.0,0.75,0.8), 'black': (0,0,0), 'white': (1,1,1),
        'gray': (0.5,0.5,0.5), 'brown': (0.6,0.4,0.2)
    }
    query_lower = query.lower()
    query_colors = [rgb for name,rgb in COLOR_MAP.items() if name in query_lower]

    query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
    if query_emb is None or search_engine.index is None:
        return {"status": "error", "message": "No images indexed. Run build_index.py first."}

    total = search_engine.index.ntotal
    q = query_emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(q)
    distances, indices = search_engine.index.search(q, total)

    db = SessionLocal()
    try:
        query_words = processed_query.lower().split()
        sig_words   = [w for w in query_words if len(w) > 2]
        all_candidates = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            clip_score = float(dist)
            # Generous floor — just cut obvious noise
            if clip_score < CLIP_SCORE_MIN:
                break   # FAISS is sorted; everything below is worse

            img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
            if not img:
                continue

            # OCR bonus
            ocr_bonus = 0.0
            if sig_words and img.ocr_text:
                txt = img.ocr_text.lower()
                hits = sum(1 for w in sig_words if w in txt)
                ocr_bonus = min(hits / len(sig_words), 1.0)

            # Tag bonus — exact word, no substrings
            tag_bonus = 0.0
            if img.scene_label:
                tag_words = {w for tag in img.scene_label.split(",")
                             for w in tag.strip().lower().split()}
                if any(w in tag_words for w in sig_words):
                    tag_bonus = 1.0

            # ===== COLOR BONUS =====
            color_bonus = 0.0
            if query_colors and getattr(img, 'avg_r', None) is not None:
                img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
                for qc in query_colors:
                    dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
                    score = max(0.0, 1.0 - dist_color / np.sqrt(3))
                    color_bonus = max(color_bonus, score)

            # CLIP dominates; OCR/tag only break ties. Color is weighted if present.
            final_score = 0.70 * clip_score + 0.15 * ocr_bonus + 0.10 * color_bonus + 0.05 * tag_bonus

            all_candidates.append({
                "img": img,
                "clip": clip_score,
                "final": final_score,
            })

        if not all_candidates:
            return {"status": "not_found", "message": f"No images matched '{query}'."}

        # ── Gap detection ────────────────────────────────────────────────────
        # Sort high → low, find the largest drop between consecutive scores.
        # That drop separates relevant from irrelevant.
        all_candidates.sort(key=lambda x: x["final"], reverse=True)
        finals = [c["final"] for c in all_candidates]

        cut_idx = len(finals)   # default: keep everything
        if len(finals) >= 3:
            max_gap   = 0.0
            min_gap_required = 0.025  # at least 2.5-point drop to bother cutting
            for i in range(1, len(finals)):
                gap = finals[i - 1] - finals[i]
                if gap > max_gap and gap >= min_gap_required:
                    max_gap = gap
                    cut_idx = i   # cut BEFORE index i
            logger.info(f"  Gap detection: best gap={max_gap:.4f} → keeping {cut_idx}/{len(finals)}")

        kept = all_candidates[:cut_idx]
        
        # Apply dynamic threshold safety net to drop unrelated low scores
        best_score = kept[0]["final"] if kept else 0.0
        dynamic_min = best_score * 0.8
        filtered_kept = [c for c in kept if c["final"] >= dynamic_min]
        kept = filtered_kept
        
        if not kept:
            return {"status": "not_found", "message": f"No images matched '{query}'."}
        
        logger.info(f"✅ {len(kept)} results for '{query}' "
                    f"(top score={best_score:.3f}, "
                    f"cutoff={kept[-1]['final']:.3f})")

        results = []
        for c in kept[:top_k]:
            img = c["img"]
            results.append({
                "id":           img.id,
                "filename":     img.filename,
                "score":        round(c["final"] * 100, 2),
                "timestamp":    img.timestamp.isoformat() if img.timestamp else None,
                "location":     {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
                "person_count": img.person_count or 0,
            })

        if not results:
            return {"status": "not_found", "message": f"No images matched '{query}'."}

        return {"status": "found", "query": query, "count": len(results), "results": results}
    finally:
        db.close()

@app.post("/search/voice")
def voice_search(duration: int = Form(5)):
    try:
        transcribed = voice_engine.listen_and_transcribe(duration=duration)
        if not transcribed:
            return {"status": "error", "message": "No audio"}
        return search(query=transcribed, top_k=20)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/timeline")
def get_timeline():
    db = SessionLocal()
    try:
        images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
        results = [{"id": img.id, "filename": img.filename, "date": img.timestamp.isoformat() if img.timestamp else None, "thumbnail": f"/images/{img.filename}"} for img in images]
        return {"count": len(results), "results": results}
    finally:
        db.close()

@app.get("/faces")
def get_faces(person_id: int = Query(None)):
    db = SessionLocal()
    try:
        if person_id:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise HTTPException(status_code=404)
            faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
            images = []
            cover = None
            seen_image_ids = set()
            for f in faces:
                # Prefer relationship, fall back to manual lookup
                img = f.image if hasattr(f, 'image') and f.image else None
                if img is None and f.image_id:
                    img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
                if img and img.filename and img.id not in seen_image_ids:
                    seen_image_ids.add(img.id)
                    if cover is None:
                        cover = f"/images/{img.filename}"
                    images.append({
                        "id":        img.id,
                        "filename":  img.filename,
                        "thumbnail": f"/images/{img.filename}",
                        "date":      img.timestamp.isoformat() if img.timestamp else None,
                    })
            return {
                "id": person.id, "name": person.name,
                "face_count": len(faces), "cover": cover,
                "images": images, "results": images,  # "results" alias for frontend compat
            }
        else:
            people = db.query(Person).all()
            results = []
            for p in people:
                faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
                if not faces:
                    continue
                cover = None
                count = 0
                for f in faces:
                    img = f.image if hasattr(f, 'image') and f.image else None
                    if img is None and f.image_id:
                        img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
                    if img and img.filename:
                        count += 1
                        if cover is None:
                            cover = f"/images/{img.filename}"
                if cover is None:
                    continue   # skip people with no linked images at all
                results.append({"id": p.id, "name": p.name, "count": count, "cover": cover})
            return {"results": results, "count": len(results)}
    finally:
        db.close()

@app.get("/people/{person_id}")
def get_person(person_id: int):
    """Get a single person with all their images."""
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=404)
        faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
        images = []
        cover = None
        seen_image_ids = set()
        for f in faces:
            img = f.image if hasattr(f, 'image') and f.image else None
            if img is None and f.image_id:
                img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
            if img and img.filename and img.id not in seen_image_ids:
                seen_image_ids.add(img.id)
                if cover is None:
                    cover = f"/images/{img.filename}"
                images.append({
                    "id":        img.id,
                    "filename":  img.filename,
                    "thumbnail": f"/images/{img.filename}",
                    "date":      img.timestamp.isoformat() if img.timestamp else None,
                })
        
        # Only compute the real valid images found internally for frontend rendering
        return {
            "id": person.id, "name": person.name,
            "face_count": len(images), "cover": cover,
            "images": images, "results": images,
        }
    finally:
        db.close()

@app.post("/people/{person_id}")
def update_person(person_id: int, name: str = Form(...)):
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=404)
        person.name = name
        db.commit()
        return {"status": "success", "id": person.id, "name": person.name}
    finally:
        db.close()

@app.get("/people/{person_id}/celebcheck")
def check_celebrity_match(person_id: int):
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            return {"status": "no_match"}
        return {"status": "no_match"}
    finally:
        db.close()

@app.get("/albums/{album_id}")
def get_album_by_id(album_id: int):
    """Get a single album by path param."""
    db = SessionLocal()
    try:
        album = db.query(Album).filter(Album.id == album_id).first()
        if not album:
            raise HTTPException(status_code=404)
        images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
        cover = images[0].filename if images else None
        date_str = ""
        if album.start_date:
            date_str = album.start_date.strftime("%b %Y")
            if album.end_date and album.end_date.month != album.start_date.month:
                date_str += f" – {album.end_date.strftime('%b %Y')}"
        img_list = [{"id": img.id, "filename": img.filename,
                     "thumbnail": f"/images/{img.filename}",
                     "date": img.timestamp.isoformat() if img.timestamp else None}
                    for img in images]
        return {
            "id": album.id, "title": album.title, "type": album.type,
            "description": album.description, "date": date_str, "cover": cover,
            "start_date": album.start_date.isoformat() if album.start_date else None,
            "end_date":   album.end_date.isoformat()   if album.end_date   else None,
            "image_count": len(images),
            "images": img_list,
            "results": img_list,   # alias in case frontend uses "results"
        }
    finally:
        db.close()

@app.get("/albums")
def get_albums(album_id: int = Query(None)):
    db = SessionLocal()
    try:
        if album_id:
            return get_album_by_id(album_id)
        albums = db.query(Album).all()
        results = []
        for a in albums:
            album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
            if not album_images:
                continue   # Skip orphaned/empty albums — they show blank cards
            cover = album_images[0].filename
            date_str = ""
            if a.start_date:
                date_str = a.start_date.strftime("%b %Y")
                if a.end_date and a.end_date.month != a.start_date.month:
                    date_str += f" – {a.end_date.strftime('%b %Y')}"
            results.append({
                "id": a.id, "title": a.title, "type": a.type,
                "description": a.description, "date": date_str,
                "cover": cover, "count": len(album_images),
                "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]],
            })
        return {"results": results, "count": len(results)}
    finally:
        db.close()

@app.post("/favorites")
def add_favorite(image_id: int = Form(...)):
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        if not img:
            raise HTTPException(status_code=404)
        img.is_favorite = not getattr(img, 'is_favorite', False)
        db.commit()
        return {"status": "success"}
    finally:
        db.close()

@app.get("/favorites")
def get_favorites():
    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(DBImage.is_favorite == True).all()
        return {"count": len(images), "results": [{"id": img.id, "filename": img.filename} for img in images]}
    finally:
        db.close()

@app.get("/duplicates")
def get_duplicates():
    db = SessionLocal()
    try:
        all_images = db.query(DBImage).all()
        if not all_images:
            return {"status": "found", "duplicate_groups": [], "total_groups": 0}

        groups = duplicate_engine.find_duplicates_fast(all_images, hamming_threshold=5)

        # Format response — include thumbnail paths
        formatted = []
        for g in groups:
            formatted.append({
                "count": g["count"],
                "total_size": g["total_size"],
                "images": [
                    {
                        "id":        img["id"],
                        "filename":  img["filename"],
                        "thumbnail": f"/images/{img['filename']}",
                        "size":      img["size"],
                    }
                    for img in g["images"]
                ],
            })

        return {
            "status":           "found",
            "duplicate_groups": formatted,
            "total_groups":     len(formatted),
        }
    except Exception as e:
        logger.error(f"Duplicates error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "duplicate_groups": [], "total_groups": 0}
    finally:
        db.close()

@app.post("/recluster")
def recluster():
    db = SessionLocal()
    try:
        logger.info("🔄 Recluster: clearing old assignments…")
        db.query(DBFace).update({"person_id": None})
        db.query(Person).delete()
        db.query(DBImage).update({"album_id": None})
        db.query(Album).filter(Album.type == "event").delete()
        db.commit()

        # ── Face clustering ──────────────────────────────────────────────────
        face_records = db.query(DBFace).filter(DBFace.face_embedding != None).all()
        embeddings = []
        valid_face_records = []
        for fr in face_records:
            try:
                emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
                if emb.shape[0] == 512:
                    embeddings.append(emb)
                    valid_face_records.append(fr)
            except Exception as e:
                logger.warning(f"Bad embedding face {fr.id}: {e}")

        people_count = 0
        if embeddings:
            logger.info(f"👥 Clustering {len(embeddings)} face embeddings…")
            labels = face_engine.cluster_faces(embeddings)
            person_map = {}
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                if label not in person_map:
                    p = Person(name=f"Person {label + 1}")
                    db.add(p)
                    db.flush()
                    person_map[label] = p.id
                    people_count += 1
                valid_face_records[i].person_id = person_map[label]
            db.commit()
            logger.info(f"✅ {people_count} people")
        else:
            logger.warning("⚠️  No face embeddings — run build_index.py first")

        # ── Album / event detection ──────────────────────────────────────────
        all_images = db.query(DBImage).all()
        albums_count = 0
        if all_images:
            metadata = [
                {"id": img.id, "lat": img.lat or 0.0, "lon": img.lon or 0.0, "timestamp": img.timestamp}
                for img in all_images if img.timestamp
            ]
            if metadata:
                album_labels = clustering_engine.detect_events(metadata)
                album_map = {}
                for i, label in enumerate(album_labels):
                    if label == -1:
                        continue
                    if label not in album_map:
                        cluster_meta = [metadata[j] for j, l in enumerate(album_labels) if l == label]
                        ts_list = [m["timestamp"] for m in cluster_meta if m["timestamp"]]
                        start_d = min(ts_list) if ts_list else None
                        end_d   = max(ts_list) if ts_list else None
                        if start_d:
                            if end_d and end_d.date() != start_d.date():
                                title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%b %d, %Y')}"
                            else:
                                title = start_d.strftime("%b %d, %Y")
                        else:
                            title = f"Event {label + 1}"
                        new_album = Album(title=title, type="event", start_date=start_d, end_date=end_d)
                        db.add(new_album)
                        db.flush()
                        album_map[label] = new_album.id
                        albums_count += 1
                    img_id = metadata[i]["id"]
                    db.query(DBImage).filter(DBImage.id == img_id).update({"album_id": album_map[label]})
                db.commit()
                logger.info(f"✅ {albums_count} albums")

        return {"status": "done", "people": people_count, "albums": albums_count}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Recluster failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400)
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(IMAGE_DIR, filename)
    db = SessionLocal()
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        from PIL import Image as PILImage
        import json as _json
        width = height = None
        avg_r = avg_g = avg_b = 0.0
        try:
            img_pil = PILImage.open(file_path).convert("RGB")
            width, height = img_pil.size
            arr = np.array(img_pil)
            avg_r, avg_g, avg_b = float(arr[:,:,0].mean()), float(arr[:,:,1].mean()), float(arr[:,:,2].mean())
        except Exception: pass
        clip_emb = None
        try: clip_emb = search_engine.get_image_embedding(file_path)
        except Exception: pass
        ocr_text = ""
        try: ocr_text = extract_text(file_path)
        except Exception: pass
        scene_label = ""
        person_count = 0
        try:
            objects = detector_engine.detect_objects(file_path, threshold=0.5)
            scene_label = ", ".join(objects) if objects else ""
            person_count = sum(1 for o in objects if o == "person")
        except Exception:
            try: person_count = detector_engine.detect_persons(file_path)
            except Exception: pass
        img_record = DBImage(
            filename=filename, original_path=file_path, timestamp=datetime.now(),
            ocr_text=ocr_text, scene_label=scene_label, person_count=person_count,
            width=width, height=height, avg_r=avg_r, avg_g=avg_g, avg_b=avg_b,
        )
        db.add(img_record)
        db.flush()
        if clip_emb is not None:
            try:
                if search_engine.index is None:
                    search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
                new_vec = clip_emb.reshape(1, -1).astype('float32')
                faiss.normalize_L2(new_vec)
                search_engine.index.add_with_ids(new_vec, np.array([img_record.id]).astype('int64'))
                faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
            except Exception as e: logger.warning(f"FAISS update failed: {e}")
        face_count = 0
        try:
            faces = face_engine.detect_faces(file_path)
            for face in faces:
                emb = face["embedding"].astype(np.float32)
                db.add(DBFace(image_id=img_record.id, bbox=_json.dumps(face["bbox"]), face_embedding=emb.tobytes()))
                face_count += 1
        except Exception as e: logger.warning(f"Face detection failed: {e}")
        db.commit()
        if background_tasks is not None:
            should_trigger_recluster(background_tasks)
        return {"status": "success", "id": img_record.id, "filename": filename, "person_count": person_count, "face_count": face_count}
    except Exception as e:
        db.rollback()
        if os.path.exists(file_path): os.remove(file_path)
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500)
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)