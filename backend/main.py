# FIXED main.py - Complete working version with all endpoints
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil
import numpy as np
import faiss
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

from database import SessionLocal, Image as DBImage, Face as DBFace, Person, Album, init_db
from search_engine import search_engine, resolve_query
from voice_engine import voice_engine
from face_engine import face_engine
from ocr_engine import extract_text
from detector_engine import detector_engine
from duplicate_engine import duplicate_engine

# Paths
IMAGE_DIR = "../data/images"
FAISS_INDEX_PATH = "../data/index.faiss"
# Configurable thresholds and options
FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.75))
FACE_MATCH_NEIGHBORS = int(os.environ.get("FACE_MATCH_NEIGHBORS", 5))
FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
RECLUSTER_ON_UPLOAD = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
RECLUSTER_BATCH_SIZE = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))
# Search thresholds: minimum CLIP similarity and final composite score
# raised defaults to reduce irrelevant matches; can still be tuned via env vars
CLIP_SCORE_MIN = float(os.environ.get("CLIP_SCORE_MIN", 0.55))  # Minimum normalized CLIP score (0-1)
FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.40))  # Minimum final composite score
SEARCH_SCORE_THRESHOLD = float(os.environ.get("SEARCH_SCORE_THRESHOLD", 0.25))
RECLUSTER_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
recluster_last_triggered = None  # Track last recluster time for debouncing

def should_trigger_recluster(background_tasks):
    """Check if batch size reached or timer expired; schedule recluster if needed."""
    global recluster_last_triggered
    
    if not RECLUSTER_ON_UPLOAD or not background_tasks:
        return
    
    try:
        # Increment counter
        counter = 0
        if os.path.exists(RECLUSTER_COUNTER_PATH):
            try:
                with open(RECLUSTER_COUNTER_PATH, 'r') as f:
                    counter = int(f.read().strip())
            except: pass
        
        counter += 1
        with open(RECLUSTER_COUNTER_PATH, 'w') as f:
            f.write(str(counter))
        
        # Check if batch size reached
        should_trigger = counter >= RECLUSTER_BATCH_SIZE
        
        # Also check if timer expired since last recluster
        now = datetime.datetime.now()
        if recluster_last_triggered:
            elapsed = (now - recluster_last_triggered).total_seconds()
            if elapsed >= RECLUSTER_TIMER_SECONDS:
                should_trigger = True
        elif counter > 0:
            # First upload in potential batch; will trigger if timer or batch size reached
            should_trigger = counter >= RECLUSTER_BATCH_SIZE
        
        if should_trigger:
            logger.info(f"📊 Recluster triggered: counter={counter}, batch_size={RECLUSTER_BATCH_SIZE}")
            background_tasks.add_task(recluster)
            recluster_last_triggered = now
            # Reset counter
            with open(RECLUSTER_COUNTER_PATH, 'w') as f:
                f.write('0')
    except Exception as e:
        logger.warning(f"Batched recluster check failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB
    init_db()
    logger.info("🚀 Starting Offline Smart Gallery Backend...")

    # Load image FAISS index
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"✅ Image FAISS index loaded ({search_engine.index.ntotal} vectors).")
        except Exception as e:
            logger.error(f"❌ Error loading image FAISS index: {e}")
            search_engine.index = None
    else:
        logger.warning("⚠️  Image FAISS index not found. Searching will be limited.")

    # Load face FAISS index (handled by face_engine internally)
    logger.info(f"✅ Face FAISS index ready ({face_engine.face_index.ntotal if face_engine.face_index else 0} vectors).")
    yield

app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

@app.get("/health")
def health():
    return {
        "status": "ready",
        "models": ["CLIP", "OCR", "FaceRecognition", "Clustering"],
        "image_index": search_engine.index.ntotal if search_engine.index else 0,
        "face_index": face_engine.face_index.ntotal if face_engine.face_index else 0
    }

@app.get("/test-db")
def test_db():
    """Diagnostic endpoint to test database"""
    db = SessionLocal()
    try:
        count = db.query(DBImage).count()
        images = db.query(DBImage).limit(1).all()
        return {
            "status": "ok",
            "total_images": count,
            "sample": {
                "filename": images[0].filename,
                "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None
            } if images else None
        }
    except Exception as e:
        logger.error(f"❌ Test DB error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@app.post("/search")
def search(query: str = Form(...), top_k: int = Form(20)):
    """
    Search endpoint with semantic, OCR and color support.
    Returns top_k results even if scores are low (threshold only for logging).
    """

    if not query or not query.strip():
        return {"status": "error", "message": "Query cannot be empty"}

    processed_query = resolve_query(query)
    logger.info(f"🔍 Search: original='{query}' expanded='{processed_query}' threshold={SEARCH_SCORE_THRESHOLD}")

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
        return {
            "status": "error",
            "message": "No images indexed yet. Please run build_index.py and upload photos first!"
        }

    # Search FAISS - focus on top matches only
    candidate_k = min(top_k * 8, 250)  # Smaller pool focused on best matches only
    query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_emb_reshaped)
    distances, indices = search_engine.index.search(query_emb_reshaped, candidate_k)

    db = SessionLocal()
    results = []

    try:
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
            if not img:
                logger.debug(f"Image with ID {idx} not found in database")
                continue

            # FAISS returns inner product similarity for normalized vectors (range roughly [-1,1]).
            # Map to [0,1] so our hybrid ranking function expects normalized inputs.
            raw_sim = float(dist)
            clip_score = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))
            
            # CLIP score filtering: skip images with low CLIP similarity
            if clip_score < CLIP_SCORE_MIN:
                logger.debug(f"Skip {img.filename}: CLIP score {clip_score:.3f} below minimum {CLIP_SCORE_MIN}")
                continue

            # OCR matching: use the processed (expanded) query for better recall
            ocr_text = (img.ocr_text or "").lower()
            query_words = processed_query.lower().split()
            significant_words = [w for w in query_words if len(w) > 2]
            if significant_words:
                matches = sum(1 for w in significant_words if w in ocr_text)
                ocr_bonus = min(matches / len(significant_words), 1.0) if significant_words else 0.0
            else:
                ocr_bonus = 0.0

            # color bonus (unchanged)
            color_bonus = 0.0
            if query_colors and getattr(img, 'avg_r', None) is not None:
                img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
                for qc in query_colors:
                    dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
                    score = max(0.0, 1.0 - dist_color / np.sqrt(3))
                    color_bonus = max(color_bonus, score)

            # tag/object bonus: check scene_label field saved during indexing
            tag_bonus = 0.0
            tags = []  # Initialize tags list (empty if no scene_label)
            if img.scene_label:
                tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
                # if any of the processed query words match a tag we give a full bonus
                if any(w in tags for w in query_words):
                    tag_bonus = 1.0

            # require object match when query contains a known object term
            # using detector categories from detector_engine if available
            DETECT_CATS = set()
            try:
                if hasattr(detector_engine, 'categories') and detector_engine.categories:
                    DETECT_CATS = set(c.lower() for c in detector_engine.categories)
            except:
                pass
            
            object_terms = set(w for w in query_words if w in DETECT_CATS)
            if object_terms and tags:
                # if query is object-specific but no matching tag, skip result
                if not any(t in object_terms for t in tags):
                    logger.debug(f"Skip {img.filename}: no object tag for terms {object_terms}")
                    continue

            # Use centralized hybrid ranking (clips, ocr, color, tags)
            final_score = search_engine.hybrid_rank(clip_score, ocr_bonus=ocr_bonus, color_bonus=color_bonus, tag_bonus=tag_bonus)
            
            # Filter by minimum final score
            if final_score < FINAL_SCORE_MIN:
                logger.debug(f"Skip {img.filename}: final score {final_score:.3f} below minimum {FINAL_SCORE_MIN}")
                continue
            
            logger.debug(f"Match: {img.filename}, RAW_SIM={raw_sim:.3f}, CLIP_NORM={clip_score:.3f}, OCR={ocr_bonus:.2f}, COLOR={color_bonus:.2f}, Final={final_score:.3f}")

            results.append({
                "id": img.id,
                "filename": img.filename,
                "score": round(final_score * 100, 2),
                "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
                "person_count": img.person_count or 0
            })

        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        if not results:
            return {
                "status": "not_found",
                "message": f"No images found matching '{query}'",
                "suggestion": "Try more specific keywords like 'dog', 'beach', 'sunset'"
            }

        logger.info(f"✅ Found {len(results)} results for '{query}'")
        return {
            "status": "found",
            "query": query,
            "count": len(results),
            "results": results
        }
    finally:
        db.close()

@app.post("/search/voice")
def voice_search(duration: int = Form(5)):
    """Search using voice input"""
    try:
        transcribed = voice_engine.listen_and_transcribe(duration=duration)
        if not transcribed:
            return {"status": "error", "message": "Could not transcribe audio"}
        
        # Perform search with transcribed text
        return search(query=transcribed, top_k=20)
    except Exception as e:
        logger.error(f"Voice search failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/timeline")
def get_timeline():
    """Get all images organized chronologically"""
    db = SessionLocal()
    try:
        images = db.query(DBImage).order_by(DBImage.timestamp.desc()).all()
        
        results = []
        for img in images:
            results.append({
                "id": img.id,
                "filename": img.filename,
                "date": img.timestamp.isoformat() if img.timestamp else None,
                "thumbnail": f"/images/{img.filename}"
            })
        
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.error(f"❌ Timeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/faces")
def get_faces(person_id: int = Query(None)):
    """Get detected people and their face counts"""
    db = SessionLocal()
    try:
        if person_id:
            # Get specific person's faces
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise HTTPException(status_code=404, detail="Person not found")
            
            faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
            images = []
            for face in faces:
                img = db.query(DBImage).filter(DBImage.id == face.image_id).first()
                if img:
                    images.append({
                        "id": img.id,
                        "filename": img.filename,
                        "thumbnail": f"/images/{img.filename}",
                        "date": img.timestamp.isoformat() if img.timestamp else None
                    })
            
            return {
                "id": person.id,
                "name": person.name,
                "face_count": len(faces),
                "images": images
            }
        else:
            # Get all people
            people = db.query(Person).all()
            results = []
            for p in people:
                faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
                
                if not faces:
                    continue  # Skip people with no faces
                
                # Get all images for this person (not just cover)
                images = []
                cover_filename = None
                for face in faces:
                    img = db.query(DBImage).filter(DBImage.id == face.image_id).first()
                    if img:
                        if not cover_filename:
                            cover_filename = img.filename
                        images.append({
                            "id": img.id,
                            "filename": img.filename,
                            "thumbnail": f"/images/{img.filename}",
                            "date": img.timestamp.isoformat() if img.timestamp else None
                        })
                
                results.append({
                    "id": p.id,
                    "name": p.name,
                    "count": len(faces),
                    "cover": f"/images/{cover_filename}" if cover_filename else None,
                    "images": images
                })
            
            return {"results": results, "count": len(results)}
    finally:
        db.close()

@app.post("/people/{person_id}")
def update_person(person_id: int, name: str = Form(...)):
    """Rename a person"""
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        person.name = name
        db.commit()
        logger.info(f"✅ Renamed person {person_id} to '{name}'")
        
        return {
            "status": "success",
            "id": person.id,
            "name": person.name
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Update person failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/people/{person_id}/celebcheck")
def check_celebrity_match(person_id: int):
    """Try to identify if person matches a known celebrity (uses Google reverse image search hint)"""
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Get first face image of this person
        faces = db.query(DBFace).filter(DBFace.person_id == person_id).limit(1).all()
        if not faces or not faces[0].image_id:
            return {
                "status": "no_match",
                "message": "No face found for this person"
            }
        
        img = db.query(DBImage).filter(DBImage.id == faces[0].image_id).first()
        if not img or not os.path.exists(img.original_path):
            return {
                "status": "no_match",
                "message": "Face image not found"
            }
        
        # Try lightweight celeb detection (simple name suggestion based on OCR context)
        # In a real system, you'd call Google Reverse Image Search or Celebrity API
        ocr_text = img.ocr_text or ""
        if ocr_text:
            # Extract potential names (all caps words or words starting with capitals)
            import re
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', ocr_text)
            if names:
                # Return first few potential names
                logger.info(f"Suggested names from OCR: {names}")
                return {
                    "status": "suggestions",
                    "suggestions": names[:3]
                }
        
        # No OCR context found
        return {
            "status": "no_match",
            "message": "Could not identify from available context. Try manual entry or Google Images.",
            "suggestion": "You can manually edit the name to identify this person"
        }
    except Exception as e:
        logger.error(f"Celebrity check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        db.close()

@app.get("/albums")
def get_albums(album_id: int = Query(None)):
    """Get auto-generated albums (trips/events)"""
    db = SessionLocal()
    try:
        if album_id:
            # Get specific album
            album = db.query(Album).filter(Album.id == album_id).first()
            if not album:
                raise HTTPException(status_code=404, detail="Album not found")
            
            images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
            
            image_list = []
            for img in images:
                image_list.append({
                    "id": img.id,
                    "filename": img.filename,
                    "date": img.timestamp.isoformat() if img.timestamp else None,
                    "thumbnail": f"/images/{img.filename}"
                })
            
            return {
                "id": album.id,
                "title": album.title,
                "type": album.type,
                "description": album.description,
                "start_date": album.start_date.isoformat() if album.start_date else None,
                "end_date": album.end_date.isoformat() if album.end_date else None,
                "image_count": len(images),
                "images": image_list
            }
        else:
            # Get all albums
            albums = db.query(Album).all()
            results = []
            
            for a in albums:
                album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
                cover = f"/images/{album_images[0].filename}" if album_images else None
                
                # Format date range nicely
                date_str = ""
                if a.start_date:
                    date_str = a.start_date.strftime("%b %Y")
                    if a.end_date and a.end_date.month != a.start_date.month:
                        date_str += f" – {a.end_date.strftime('%b %Y')}"
                
                results.append({
                    "id": a.id,
                    "title": a.title,
                    "type": a.type,
                    "cover": cover,
                    "count": len(album_images),
                    "date": date_str,
                    "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]]
                })
            
            return {"results": results, "count": len(results)}
    finally:
        db.close()

@app.post("/favorites")
def add_favorite(image_id: int = Form(...)):
    """Mark image as favorite or toggle favorite status"""
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Toggle favorite status
        img.is_favorite = not getattr(img, 'is_favorite', False)
        db.commit()
        
        logger.info(f"Image {image_id} favorite toggled to {img.is_favorite}")
        
        return {
            "status": "success",
            "image_id": image_id,
            "is_favorite": img.is_favorite
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Add favorite failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/favorites")
def get_favorites():
    """Get all favorite images"""
    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(DBImage.is_favorite == True).order_by(DBImage.timestamp.desc()).all()
        
        results = []
        for img in images:
            results.append({
                "id": img.id,
                "filename": img.filename,
                "date": img.timestamp.isoformat() if img.timestamp else None,
                "thumbnail": f"/images/{img.filename}"
            })
        
        logger.info(f"Retrieved {len(results)} favorite images")
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Get favorites error: {e}")
        return {"count": 0, "results": []}
    finally:
        db.close()

@app.get("/duplicates")
def get_duplicates():
    """Find and list duplicate images using perceptual hashing"""
    db = SessionLocal()
    try:
        all_images = db.query(DBImage).all()
        
        duplicate_groups = []
        processed = set()
        
        logger.info(f"Scanning {len(all_images)} images for duplicates...")
        
        for i, img1 in enumerate(all_images):
            if img1.id in processed:
                continue
            
            if not img1.original_path or not os.path.exists(img1.original_path):
                continue
            
            duplicates_of_this = [img1]
            
            # Compare with remaining images
            for img2 in all_images[i+1:]:
                if img2.id in processed:
                    continue
                
                if not img2.original_path or not os.path.exists(img2.original_path):
                    continue
                
                # Check perceptual hash similarity
                hash1 = duplicate_engine.get_phash(img1.original_path)
                hash2 = duplicate_engine.get_phash(img2.original_path)
                
                if hash1 and hash2:
                    # Hamming distance < 5 bits = likely duplicate
                    diff = bin(hash1 ^ hash2).count('1')
                    if diff < 5:
                        duplicates_of_this.append(img2)
                        processed.add(img2.id)
            
            # Only return groups with duplicates
            if len(duplicates_of_this) > 1:
                group = []
                for img in duplicates_of_this:
                    size = os.path.getsize(img.original_path) if os.path.exists(img.original_path) else 0
                    group.append({
                        "id": img.id,
                        "filename": img.filename,
                        "thumbnail": f"/images/{img.filename}",
                        "size": size,
                        "date": img.timestamp.isoformat() if img.timestamp else None
                    })
                
                duplicate_groups.append({
                    "count": len(group),
                    "images": group,
                    "total_size": sum(img['size'] for img in group)
                })
                
                for img in duplicates_of_this:
                    processed.add(img.id)
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        
        return {
            "status": "found" if duplicate_groups else "not_found",
            "duplicate_groups": duplicate_groups,
            "total_groups": len(duplicate_groups),
            "total_duplicates": sum(len(g["images"]) for g in duplicate_groups),
            "potential_savings_mb": sum(g["total_size"] for g in duplicate_groups) / (1024*1024)
        }
    except Exception as e:
        logger.error(f"Duplicate detection error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "duplicate_groups": []
        }
    finally:
        db.close()

@app.post("/recluster")
def recluster():
    """Clears old auto-generated people/albums and re-runs clustering from scratch."""
    db = SessionLocal()
    try:
        logger.info("🔄 Starting recluster operation...")
        
        # ── CLEAR OLD AUTO-CLUSTERED DATA ──────────────────────────────────────
        db.query(DBFace).update({"person_id": None})
        db.query(Person).delete()
        db.query(DBImage).update({"album_id": None})
        db.query(Album).filter(Album.type == "event").delete()
        db.commit()
        logger.info("✅ Cleared old clustering data")

        # ── FACE CLUSTERING ────────────────────────────────────────────────────
        all_faces = db.query(DBFace).all()
        embeddings = []
        valid_faces = []
        
        for face in all_faces:
            if face.face_embedding is not None:
                try:
                    arr = np.frombuffer(face.face_embedding, dtype=np.float32)
                    embeddings.append(arr)
                    valid_faces.append(face)
                except Exception as e:
                    logger.warning(f"Could not parse embedding for face {face.id}: {e}")

        face_count = 0
        if embeddings:
            labels = face_engine.cluster_faces(embeddings)
            person_map = {}
            
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                
                if label not in person_map:
                    new_person = Person(name=f"Person {label + 1}")
                    db.add(new_person)
                    db.flush()
                    person_map[label] = new_person.id
                    face_count += 1
                
                valid_faces[i].person_id = person_map[label]
            
            db.commit()
            face_engine.rebuild_index(embeddings, [f.id for f in valid_faces])
            logger.info(f"✅ Clustered {len(embeddings)} faces into {face_count} people")
        else:
            logger.warning("⚠️  No face embeddings found")

        # ── ALBUM/EVENT CLUSTERING ─────────────────────────────────────────────
        from clustering_engine import clustering_engine
        all_images = db.query(DBImage).all()
        album_count = 0
        
        if all_images:
            metadata = [
                {
                    "id": img.id,
                    "lat": img.lat or 0.0,
                    "lon": img.lon or 0.0,
                    "timestamp": img.timestamp
                }
                for img in all_images if img.timestamp
            ]
            
            if metadata:
                album_labels = clustering_engine.detect_events(metadata)
                album_map = {}
                
                for i, label in enumerate(album_labels):
                    if label == -1:
                        continue
                    
                    if label not in album_map:
                        cluster_imgs = [metadata[j] for j, l in enumerate(album_labels) if l == label]
                        ts_list = [m['timestamp'] for m in cluster_imgs if m['timestamp']]
                        
                        start_d = min(ts_list) if ts_list else None
                        end_d = max(ts_list) if ts_list else None
                        
                        new_album = Album(
                            title=f"Event {label + 1}",
                            type="event",
                            start_date=start_d,
                            end_date=end_d
                        )
                        db.add(new_album)
                        db.flush()
                        album_map[label] = new_album.id
                        album_count += 1
                    
                    all_images[i].album_id = album_map[label]
                
                db.commit()
                logger.info(f"✅ Created {album_count} albums")

        return {
            "status": "done",
            "people": face_count,
            "albums": album_count
        }
    
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Recluster failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/upload")
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload and process a single image"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use JPG or PNG.")
    
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(IMAGE_DIR, filename)
    
    db = SessionLocal()
    
    try:
        # 1. Save File
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📤 Uploaded {filename}")
        
        # 2. Get image dimensions
        from PIL import Image as PILImage
        try:
            img_pil = PILImage.open(file_path)
            width, height = img_pil.size
        except:
            width, height = None, None
        
        # compute average color
        try:
            im_color = img_pil.convert('RGB') if 'img_pil' in locals() else PILImage.open(file_path).convert('RGB')
            arr = np.array(im_color, dtype=np.float32)
            avg = arr.mean(axis=(0,1))
            avg_r, avg_g, avg_b = float(avg[0]), float(avg[1]), float(avg[2])
        except Exception:
            avg_r = avg_g = avg_b = 0.0
        
        # 3. Semantic Embedding
        clip_emb = None
        try:
            clip_emb = search_engine.get_image_embedding(file_path)
            logger.info(f"✅ CLIP embedding extracted for {filename}")
        except Exception as e:
            logger.error(f"❌ CLIP embedding failed: {e}")

        # 4. OCR
        ocr_text = ""
        try:
            ocr_text = extract_text(file_path)
            logger.info(f"✅ OCR completed: {len(ocr_text)} chars")
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
        
        # 5. Person Detection
        person_count = 0
        try:
            person_count = detector_engine.detect_persons(file_path)
            logger.info(f"✅ Detected {person_count} people")
        except Exception as e:
            logger.error(f"❌ Person detection failed: {e}")
        
        # 6. Create Image Record
        img_record = DBImage(
            filename=filename,
            original_path=file_path,
            timestamp=datetime.now(),
            ocr_text=ocr_text,
            person_count=person_count,
            width=width,
            avg_r=avg_r,
            avg_g=avg_g,
            avg_b=avg_b,
            height=height,
            size_bytes=os.path.getsize(file_path)
        )
        db.add(img_record)
        db.flush()
        
        logger.info(f"✅ Created image record ID={img_record.id}")
        
        # 7. Face Detection & Indexing
        face_count = 0
        try:
            faces = face_engine.detect_faces(file_path)
            logger.info(f"✅ Detected {len(faces)} faces")
            
            for face in faces:
                emb = face['embedding'].astype(np.float32)
                emb_blob = emb.tobytes()

                # Try to match this face against existing face index
                assigned_person_id = None
                try:
                    if face_engine.face_index is not None and face_engine.face_index.ntotal > 0:
                        vec = emb.reshape(1, -1).astype('float32')
                        faiss.normalize_L2(vec)
                        # search multiple neighbors and use voting to improve accuracy
                        D, I = face_engine.face_index.search(vec, FACE_MATCH_NEIGHBORS)
                        votes = {}
                        total_sim = 0.0
                        for sim_val, idx_pos in zip(D[0], I[0]):
                            if idx_pos == -1:
                                continue
                            sim = float(sim_val)
                            total_sim += sim
                            try:
                                matched_face_db_id = face_engine.face_id_map[idx_pos]
                                matched_face = db.query(DBFace).filter(DBFace.id == int(matched_face_db_id)).first()
                                if matched_face and matched_face.person_id:
                                    pid = matched_face.person_id
                                    votes[pid] = votes.get(pid, 0.0) + sim
                            except Exception:
                                continue

                        # pick top voted person if it has majority of similarity mass
                        if votes and total_sim > 0:
                            best_pid, best_score = max(votes.items(), key=lambda x: x[1])
                            if best_score / total_sim >= FACE_MATCH_VOTE_RATIO and best_score >= FACE_MATCH_THRESHOLD:
                                assigned_person_id = best_pid
                except Exception as e:
                    logger.warning(f"Face matching check failed: {e}")

                # If no matching person found, create a new Person record
                if assigned_person_id is None:
                    try:
                        new_person = Person(name=f"Person")
                        db.add(new_person)
                        db.flush()
                        assigned_person_id = new_person.id
                    except Exception as e:
                        logger.error(f"Could not create person record: {e}")
                        assigned_person_id = None

                face_record = DBFace(
                    image_id=img_record.id,
                    bbox=str(face['bbox']),
                    face_embedding=emb_blob,
                    person_id=assigned_person_id
                )
                db.add(face_record)
                db.flush()

                # Add to face FAISS index and maintain id map
                try:
                    face_engine.add_to_index(emb, face_record.id)
                except Exception as e:
                    logger.error(f"Failed to add face to index: {e}")

                face_count += 1
                logger.info(f"✅ Processed and indexed face {face_count} (person_id={assigned_person_id})")
        
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
        
        # 8. Update Image FAISS Index
        if clip_emb is not None:
            try:
                if search_engine.index is None:
                    dim = clip_emb.shape[0]
                    sub_index = faiss.IndexFlatIP(dim)
                    search_engine.index = faiss.IndexIDMap(sub_index)
                
                new_vec = clip_emb.reshape(1, -1).astype('float32')
                faiss.normalize_L2(new_vec)
                
                ids_np = np.array([img_record.id]).astype('int64')
                search_engine.index.add_with_ids(new_vec, ids_np)
                
                faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
                logger.info(f"✅ Updated image FAISS index")
            except Exception as e:
                logger.error(f"❌ Index update failed: {e}")
        
        db.commit()

        # Batched/debounced recluster: only trigger when batch size reached or timer expired
        if background_tasks is not None:
            should_trigger_recluster(background_tasks)

        return {
            "status": "success",
            "id": img_record.id,
            "filename": filename,
            "person_count": person_count,
            "face_count": face_count
        }
    
    except Exception as e:
        db.rollback()
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)