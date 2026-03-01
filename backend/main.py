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
# #     Gap-detection search.

# #     Problem with a fixed threshold: "dog" needs ≥0.29, "cat" ≥0.26, "horse" ≥0.32.
# #     No single number works for all queries.

# #     Solution: score all candidates above a generous floor (0.20), then find the
# #     biggest SCORE DROP between consecutive results (sorted high→low). Everything
# #     above that drop is relevant; everything below is noise. The gap naturally
# #     varies per query.
# #     """
# #     if not query or not query.strip():
# #         return {"status": "error", "message": "Query empty"}

# #     processed_query = resolve_query(query)
# #     logger.info(f"🔍 Search: '{query}' → '{processed_query}'")

# #     # extract color words from query for color bonus
# #     COLOR_MAP = {
# #         'red': (1.0,0,0), 'blue': (0,0,1.0), 'green': (0,1.0,0),
# #         'yellow': (1.0,1.0,0), 'orange': (1.0,0.5,0), 'purple': (0.5,0,0.5),
# #         'pink': (1.0,0.75,0.8), 'black': (0,0,0), 'white': (1,1,1),
# #         'gray': (0.5,0.5,0.5), 'brown': (0.6,0.4,0.2)
# #     }
# #     query_lower = query.lower()
# #     query_colors = [rgb for name,rgb in COLOR_MAP.items() if name in query_lower]

# #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# #     if query_emb is None or search_engine.index is None:
# #         return {"status": "error", "message": "No images indexed. Run build_index.py first."}

# #     total = search_engine.index.ntotal
# #     q = query_emb.reshape(1, -1).astype('float32')
# #     faiss.normalize_L2(q)
# #     distances, indices = search_engine.index.search(q, total)

# #     db = SessionLocal()
# #     try:
# #         query_words = processed_query.lower().split()
# #         sig_words   = [w for w in query_words if len(w) > 2]
# #         all_candidates = []

# #         for dist, idx in zip(distances[0], indices[0]):
# #             if idx == -1:
# #                 continue
# #             clip_score = float(dist)
# #             # Generous floor — just cut obvious noise
# #             if clip_score < CLIP_SCORE_MIN:
# #                 break   # FAISS is sorted; everything below is worse

# #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# #             if not img:
# #                 continue

# #             # OCR bonus
# #             ocr_bonus = 0.0
# #             if sig_words and img.ocr_text:
# #                 txt = img.ocr_text.lower()
# #                 hits = sum(1 for w in sig_words if w in txt)
# #                 ocr_bonus = min(hits / len(sig_words), 1.0)

# #             # Tag bonus — exact word, no substrings
# #             tag_bonus = 0.0
# #             if img.scene_label:
# #                 tag_words = {w for tag in img.scene_label.split(",")
# #                              for w in tag.strip().lower().split()}
# #                 if any(w in tag_words for w in sig_words):
# #                     tag_bonus = 1.0

# #             # ===== COLOR BONUS =====
# #             color_bonus = 0.0
# #             if query_colors and getattr(img, 'avg_r', None) is not None:
# #                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# #                 for qc in query_colors:
# #                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# #                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
# #                     color_bonus = max(color_bonus, score)

# #             # CLIP dominates; OCR/tag only break ties. Color is weighted if present.
# #             final_score = 0.70 * clip_score + 0.15 * ocr_bonus + 0.10 * color_bonus + 0.05 * tag_bonus

# #             all_candidates.append({
# #                 "img": img,
# #                 "clip": clip_score,
# #                 "final": final_score,
# #             })

# #         if not all_candidates:
# #             return {"status": "not_found", "message": f"No images matched '{query}'."}

# #         # ── Gap detection ────────────────────────────────────────────────────
# #         # Sort high → low, find the largest drop between consecutive scores.
# #         # That drop separates relevant from irrelevant.
# #         all_candidates.sort(key=lambda x: x["final"], reverse=True)
# #         finals = [c["final"] for c in all_candidates]

# #         cut_idx = len(finals)   # default: keep everything
# #         if len(finals) >= 3:
# #             max_gap   = 0.0
# #             min_gap_required = 0.025  # at least 2.5-point drop to bother cutting
# #             for i in range(1, len(finals)):
# #                 gap = finals[i - 1] - finals[i]
# #                 if gap > max_gap and gap >= min_gap_required:
# #                     max_gap = gap
# #                     cut_idx = i   # cut BEFORE index i
# #             logger.info(f"  Gap detection: best gap={max_gap:.4f} → keeping {cut_idx}/{len(finals)}")

# #         kept = all_candidates[:cut_idx]
        
# #         # Apply dynamic threshold safety net to drop unrelated low scores
# #         best_score = kept[0]["final"] if kept else 0.0
# #         dynamic_min = best_score * 0.8
# #         filtered_kept = [c for c in kept if c["final"] >= dynamic_min]
# #         kept = filtered_kept
        
# #         if not kept:
# #             return {"status": "not_found", "message": f"No images matched '{query}'."}
        
# #         logger.info(f"✅ {len(kept)} results for '{query}' "
# #                     f"(top score={best_score:.3f}, "
# #                     f"cutoff={kept[-1]['final']:.3f})")

# #         results = []
# #         for c in kept[:top_k]:
# #             img = c["img"]
# #             results.append({
# #                 "id":           img.id,
# #                 "filename":     img.filename,
# #                 "score":        round(c["final"] * 100, 2),
# #                 "timestamp":    img.timestamp.isoformat() if img.timestamp else None,
# #                 "location":     {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# #                 "person_count": img.person_count or 0,
# #             })

# #         if not results:
# #             return {"status": "not_found", "message": f"No images matched '{query}'."}

# #         return {"status": "found", "query": query, "count": len(results), "results": results}
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
# #             seen_image_ids = set()
# #             for f in faces:
# #                 # Prefer relationship, fall back to manual lookup
# #                 img = f.image if hasattr(f, 'image') and f.image else None
# #                 if img is None and f.image_id:
# #                     img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# #                 if img and img.filename and img.id not in seen_image_ids:
# #                     seen_image_ids.add(img.id)
# #                     if cover is None:
# #                         cover = f"/images/{img.filename}"
# #                     images.append({
# #                         "id":        img.id,
# #                         "filename":  img.filename,
# #                         "thumbnail": f"/images/{img.filename}",
# #                         "date":      img.timestamp.isoformat() if img.timestamp else None,
# #                     })
# #             return {
# #                 "id": person.id, "name": person.name,
# #                 "face_count": len(faces), "cover": cover,
# #                 "images": images, "results": images,  # "results" alias for frontend compat
# #             }
# #         else:
# #             people = db.query(Person).all()
# #             results = []
# #             for p in people:
# #                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# #                 if not faces:
# #                     continue
# #                 cover = None
# #                 count = 0
# #                 for f in faces:
# #                     img = f.image if hasattr(f, 'image') and f.image else None
# #                     if img is None and f.image_id:
# #                         img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# #                     if img and img.filename:
# #                         count += 1
# #                         if cover is None:
# #                             cover = f"/images/{img.filename}"
# #                 if cover is None:
# #                     continue   # skip people with no linked images at all
# #                 results.append({"id": p.id, "name": p.name, "count": count, "cover": cover})
# #             return {"results": results, "count": len(results)}
# #     finally:
# #         db.close()

# # @app.get("/people/{person_id}")
# # def get_person(person_id: int):
# #     """Get a single person with all their images."""
# #     db = SessionLocal()
# #     try:
# #         person = db.query(Person).filter(Person.id == person_id).first()
# #         if not person:
# #             raise HTTPException(status_code=404)
# #         faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# #         images = []
# #         cover = None
# #         seen_image_ids = set()
# #         for f in faces:
# #             img = f.image if hasattr(f, 'image') and f.image else None
# #             if img is None and f.image_id:
# #                 img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
# #             if img and img.filename and img.id not in seen_image_ids:
# #                 seen_image_ids.add(img.id)
# #                 if cover is None:
# #                     cover = f"/images/{img.filename}"
# #                 images.append({
# #                     "id":        img.id,
# #                     "filename":  img.filename,
# #                     "thumbnail": f"/images/{img.filename}",
# #                     "date":      img.timestamp.isoformat() if img.timestamp else None,
# #                 })
        
# #         # Only compute the real valid images found internally for frontend rendering
# #         return {
# #             "id": person.id, "name": person.name,
# #             "face_count": len(images), "cover": cover,
# #             "images": images, "results": images,
# #         }
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

# # @app.get("/albums/{album_id}")
# # def get_album_by_id(album_id: int):
# #     """Get a single album by path param."""
# #     db = SessionLocal()
# #     try:
# #         album = db.query(Album).filter(Album.id == album_id).first()
# #         if not album:
# #             raise HTTPException(status_code=404)
# #         images = db.query(DBImage).filter(DBImage.album_id == album_id).all()
# #         cover = images[0].filename if images else None
# #         date_str = ""
# #         if album.start_date:
# #             date_str = album.start_date.strftime("%b %Y")
# #             if album.end_date and album.end_date.month != album.start_date.month:
# #                 date_str += f" – {album.end_date.strftime('%b %Y')}"
# #         img_list = [{"id": img.id, "filename": img.filename,
# #                      "thumbnail": f"/images/{img.filename}",
# #                      "date": img.timestamp.isoformat() if img.timestamp else None}
# #                     for img in images]
# #         return {
# #             "id": album.id, "title": album.title, "type": album.type,
# #             "description": album.description, "date": date_str, "cover": cover,
# #             "start_date": album.start_date.isoformat() if album.start_date else None,
# #             "end_date":   album.end_date.isoformat()   if album.end_date   else None,
# #             "image_count": len(images),
# #             "images": img_list,
# #             "results": img_list,   # alias in case frontend uses "results"
# #         }
# #     finally:
# #         db.close()

# # @app.get("/albums")
# # def get_albums(album_id: int = Query(None)):
# #     db = SessionLocal()
# #     try:
# #         if album_id:
# #             return get_album_by_id(album_id)
# #         albums = db.query(Album).all()
# #         results = []
# #         for a in albums:
# #             album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
# #             if not album_images:
# #                 continue   # Skip orphaned/empty albums — they show blank cards
# #             cover = album_images[0].filename
# #             date_str = ""
# #             if a.start_date:
# #                 date_str = a.start_date.strftime("%b %Y")
# #                 if a.end_date and a.end_date.month != a.start_date.month:
# #                     date_str += f" – {a.end_date.strftime('%b %Y')}"
# #             results.append({
# #                 "id": a.id, "title": a.title, "type": a.type,
# #                 "description": a.description, "date": date_str,
# #                 "cover": cover, "count": len(album_images),
# #                 "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]],
# #             })
# #         return {"results": results, "count": len(results)}
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
# #         if not all_images:
# #             return {"status": "found", "duplicate_groups": [], "total_groups": 0}

# #         groups = duplicate_engine.find_duplicates_fast(all_images, hamming_threshold=5)

# #         # Format response — include thumbnail paths
# #         formatted = []
# #         for g in groups:
# #             formatted.append({
# #                 "count": g["count"],
# #                 "total_size": g["total_size"],
# #                 "images": [
# #                     {
# #                         "id":        img["id"],
# #                         "filename":  img["filename"],
# #                         "thumbnail": f"/images/{img['filename']}",
# #                         "size":      img["size"],
# #                     }
# #                     for img in g["images"]
# #                 ],
# #             })

# #         return {
# #             "status":           "found",
# #             "duplicate_groups": formatted,
# #             "total_groups":     len(formatted),
# #         }
# #     except Exception as e:
# #         logger.error(f"Duplicates error: {e}", exc_info=True)
# #         return {"status": "error", "message": str(e), "duplicate_groups": [], "total_groups": 0}
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
#     Gap-detection search.

#     Problem with a fixed threshold: "dog" needs ≥0.29, "cat" ≥0.26, "horse" ≥0.32.
#     No single number works for all queries.

#     Solution: score all candidates above a generous floor (0.20), then find the
#     biggest SCORE DROP between consecutive results (sorted high→low). Everything
#     above that drop is relevant; everything below is noise. The gap naturally
#     varies per query.
#     """
#     if not query or not query.strip():
#         return {"status": "error", "message": "Query empty"}

#     processed_query = resolve_query(query)
#     logger.info(f"🔍 Search: '{query}' → '{processed_query}'")

#     # extract color words from query for color bonus
#     COLOR_MAP = {
#         'red': (1.0,0,0), 'blue': (0,0,1.0), 'green': (0,1.0,0),
#         'yellow': (1.0,1.0,0), 'orange': (1.0,0.5,0), 'purple': (0.5,0,0.5),
#         'pink': (1.0,0.75,0.8), 'black': (0,0,0), 'white': (1,1,1),
#         'gray': (0.5,0.5,0.5), 'brown': (0.6,0.4,0.2)
#     }
#     query_lower = query.lower()
#     query_colors = [rgb for name,rgb in COLOR_MAP.items() if name in query_lower]

#     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
#     if query_emb is None or search_engine.index is None:
#         return {"status": "error", "message": "No images indexed. Run build_index.py first."}

#     total = search_engine.index.ntotal
#     q = query_emb.reshape(1, -1).astype('float32')
#     faiss.normalize_L2(q)
#     distances, indices = search_engine.index.search(q, total)

#     db = SessionLocal()
#     try:
#         query_words = processed_query.lower().split()
#         sig_words   = [w for w in query_words if len(w) > 2]
#         all_candidates = []

#         for dist, idx in zip(distances[0], indices[0]):
#             if idx == -1:
#                 continue
#             clip_score = float(dist)
#             # Generous floor — just cut obvious noise
#             if clip_score < CLIP_SCORE_MIN:
#                 break   # FAISS is sorted; everything below is worse

#             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
#             if not img:
#                 continue

#             # OCR bonus
#             ocr_bonus = 0.0
#             if sig_words and img.ocr_text:
#                 txt = img.ocr_text.lower()
#                 hits = sum(1 for w in sig_words if w in txt)
#                 ocr_bonus = min(hits / len(sig_words), 1.0)

#             # Tag bonus — exact word, no substrings
#             tag_bonus = 0.0
#             if img.scene_label:
#                 tag_words = {w for tag in img.scene_label.split(",")
#                              for w in tag.strip().lower().split()}
#                 if any(w in tag_words for w in sig_words):
#                     tag_bonus = 1.0

#             # ===== COLOR BONUS =====
#             color_bonus = 0.0
#             if query_colors and getattr(img, 'avg_r', None) is not None:
#                 img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
#                 for qc in query_colors:
#                     dist_color = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
#                     score = max(0.0, 1.0 - dist_color / np.sqrt(3))
#                     color_bonus = max(color_bonus, score)

#             # CLIP dominates; OCR/tag only break ties. Color is weighted if present.
#             final_score = 0.70 * clip_score + 0.15 * ocr_bonus + 0.10 * color_bonus + 0.05 * tag_bonus

#             all_candidates.append({
#                 "img": img,
#                 "clip": clip_score,
#                 "final": final_score,
#             })

#         if not all_candidates:
#             return {"status": "not_found", "message": f"No images matched '{query}'."}

#         # ── Gap detection ────────────────────────────────────────────────────
#         # Sort high → low, find the largest drop between consecutive scores.
#         # That drop separates relevant from irrelevant.
#         all_candidates.sort(key=lambda x: x["final"], reverse=True)
#         finals = [c["final"] for c in all_candidates]

#         cut_idx = len(finals)   # default: keep everything
#         if len(finals) >= 3:
#             max_gap   = 0.0
#             min_gap_required = 0.025  # at least 2.5-point drop to bother cutting
#             for i in range(1, len(finals)):
#                 gap = finals[i - 1] - finals[i]
#                 if gap > max_gap and gap >= min_gap_required:
#                     max_gap = gap
#                     cut_idx = i   # cut BEFORE index i
#             logger.info(f"  Gap detection: best gap={max_gap:.4f} → keeping {cut_idx}/{len(finals)}")

#         kept = all_candidates[:cut_idx]
        
#         # Apply dynamic threshold safety net to drop unrelated low scores
#         best_score = kept[0]["final"] if kept else 0.0
#         dynamic_min = best_score * 0.8
#         filtered_kept = [c for c in kept if c["final"] >= dynamic_min]
#         kept = filtered_kept
        
#         if not kept:
#             return {"status": "not_found", "message": f"No images matched '{query}'."}
        
#         logger.info(f"✅ {len(kept)} results for '{query}' "
#                     f"(top score={best_score:.3f}, "
#                     f"cutoff={kept[-1]['final']:.3f})")

#         results = []
#         for c in kept[:top_k]:
#             img = c["img"]
#             results.append({
#                 "id":           img.id,
#                 "filename":     img.filename,
#                 "score":        round(c["final"] * 100, 2),
#                 "timestamp":    img.timestamp.isoformat() if img.timestamp else None,
#                 "location":     {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
#                 "person_count": img.person_count or 0,
#             })

#         if not results:
#             return {"status": "not_found", "message": f"No images matched '{query}'."}

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
#             seen_image_ids = set()
#             for f in faces:
#                 # Prefer relationship, fall back to manual lookup
#                 img = f.image if hasattr(f, 'image') and f.image else None
#                 if img is None and f.image_id:
#                     img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
#                 if img and img.filename and img.id not in seen_image_ids:
#                     seen_image_ids.add(img.id)
#                     if cover is None:
#                         cover = f"/images/{img.filename}"
#                     images.append({
#                         "id":        img.id,
#                         "filename":  img.filename,
#                         "thumbnail": f"/images/{img.filename}",
#                         "date":      img.timestamp.isoformat() if img.timestamp else None,
#                     })
#             return {
#                 "id": person.id, "name": person.name,
#                 "face_count": len(faces), "cover": cover,
#                 "images": images, "results": images,  # "results" alias for frontend compat
#             }
#         else:
#             people = db.query(Person).all()
#             results = []
#             for p in people:
#                 faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
#                 if not faces:
#                     continue
#                 cover = None
#                 count = 0
#                 for f in faces:
#                     img = f.image if hasattr(f, 'image') and f.image else None
#                     if img is None and f.image_id:
#                         img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
#                     if img and img.filename:
#                         count += 1
#                         if cover is None:
#                             cover = f"/images/{img.filename}"
#                 if cover is None:
#                     continue   # skip people with no linked images at all
#                 results.append({"id": p.id, "name": p.name, "count": count, "cover": cover})
#             return {"results": results, "count": len(results)}
#     finally:
#         db.close()

# @app.get("/people/{person_id}")
# def get_person(person_id: int):
#     """Get a single person with all their images."""
#     db = SessionLocal()
#     try:
#         person = db.query(Person).filter(Person.id == person_id).first()
#         if not person:
#             raise HTTPException(status_code=404)
#         faces = db.query(DBFace).filter(DBFace.person_id == person_id).all()
#         images = []
#         cover = None
#         seen_image_ids = set()
#         for f in faces:
#             img = f.image if hasattr(f, 'image') and f.image else None
#             if img is None and f.image_id:
#                 img = db.query(DBImage).filter(DBImage.id == f.image_id).first()
#             if img and img.filename and img.id not in seen_image_ids:
#                 seen_image_ids.add(img.id)
#                 if cover is None:
#                     cover = f"/images/{img.filename}"
#                 images.append({
#                     "id":        img.id,
#                     "filename":  img.filename,
#                     "thumbnail": f"/images/{img.filename}",
#                     "date":      img.timestamp.isoformat() if img.timestamp else None,
#                 })
        
#         # Only compute the real valid images found internally for frontend rendering
#         return {
#             "id": person.id, "name": person.name,
#             "face_count": len(images), "cover": cover,
#             "images": images, "results": images,
#         }
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
#     """Get a single album by path param."""
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
#         img_list = [{"id": img.id, "filename": img.filename,
#                      "thumbnail": f"/images/{img.filename}",
#                      "date": img.timestamp.isoformat() if img.timestamp else None}
#                     for img in images]
#         return {
#             "id": album.id, "title": album.title, "type": album.type,
#             "description": album.description, "date": date_str, "cover": cover,
#             "start_date": album.start_date.isoformat() if album.start_date else None,
#             "end_date":   album.end_date.isoformat()   if album.end_date   else None,
#             "image_count": len(images),
#             "images": img_list,
#             "results": img_list,   # alias in case frontend uses "results"
#         }
#     finally:
#         db.close()

# @app.get("/albums")
# def get_albums(album_id: int = Query(None)):
#     db = SessionLocal()
#     try:
#         if album_id:
#             return get_album_by_id(album_id)
#         albums = db.query(Album).all()
#         results = []
#         for a in albums:
#             album_images = db.query(DBImage).filter(DBImage.album_id == a.id).all()
#             if not album_images:
#                 continue   # Skip orphaned/empty albums — they show blank cards
#             cover = f"/images/{album_images[0].filename}"
#             date_str = ""
#             if a.start_date:
#                 date_str = a.start_date.strftime("%b %Y")
#                 if a.end_date and a.end_date.month != a.start_date.month:
#                     date_str += f" – {a.end_date.strftime('%b %Y')}"
#             results.append({
#                 "id": a.id, "title": a.title, "type": a.type,
#                 "description": a.description, "date": date_str,
#                 "cover": cover, "count": len(album_images),
#                 "thumbnails": [f"/images/{img.filename}" for img in album_images[:4]],
#             })
#         return {"results": results, "count": len(results)}
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
#         if not all_images:
#             return {"status": "found", "duplicate_groups": [], "total_groups": 0}

#         groups = duplicate_engine.find_duplicates_fast(all_images, hamming_threshold=5)

#         # Format response — include thumbnail paths
#         formatted = []
#         for g in groups:
#             formatted.append({
#                 "count": g["count"],
#                 "total_size": g["total_size"],
#                 "images": [
#                     {
#                         "id":        img["id"],
#                         "filename":  img["filename"],
#                         "thumbnail": f"/images/{img['filename']}",
#                         "size":      img["size"],
#                     }
#                     for img in g["images"]
#                 ],
#             })

#         return {
#             "status":           "found",
#             "duplicate_groups": formatted,
#             "total_groups":     len(formatted),
#         }
#     except Exception as e:
#         logger.error(f"Duplicates error: {e}", exc_info=True)
#         return {"status": "error", "message": str(e), "duplicate_groups": [], "total_groups": 0}
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
    return {"status": "ready", "image_index": search_engine.index.ntotal if search_engine.index else 0}

@app.get("/debug")
def debug_db():
    """
    Returns a detailed summary of what is actually in the database.
    Use this to diagnose why people/albums appear empty.
    Hit: GET http://localhost:8000/debug
    """
    db = SessionLocal()
    try:
        total_images  = db.query(DBImage).count()
        total_faces   = db.query(DBFace).count()
        faces_with_embedding = db.query(DBFace).filter(DBFace.face_embedding != None).count()
        faces_with_image_id  = db.query(DBFace).filter(DBFace.image_id != None).count()
        faces_with_person    = db.query(DBFace).filter(DBFace.person_id != None).count()
        total_people  = db.query(Person).count()
        total_albums  = db.query(Album).count()
        images_in_album = db.query(DBImage).filter(DBImage.album_id != None).count()

        people_detail = []
        for p in db.query(Person).all():
            faces = db.query(DBFace).filter(DBFace.person_id == p.id).all()
            img_ids = [f.image_id for f in faces if f.image_id]
            people_detail.append({
                "id": p.id, "name": p.name,
                "face_count": len(faces),
                "faces_with_image_id": len(img_ids),
            })

        album_detail = []
        for a in db.query(Album).all():
            imgs = db.query(DBImage).filter(DBImage.album_id == a.id).count()
            album_detail.append({"id": a.id, "title": a.title, "image_count": imgs})

        return {
            "images":  total_images,
            "faces":   total_faces,
            "faces_with_embedding": faces_with_embedding,
            "faces_with_image_id":  faces_with_image_id,
            "faces_with_person":    faces_with_person,
            "people":  total_people,
            "albums":  total_albums,
            "images_in_album": images_in_album,
            "people_detail": people_detail,
            "album_detail":  album_detail,
            "action_needed": (
                "Run POST /recluster to fix person/album assignments"
                if faces_with_person == 0 or images_in_album == 0 else "OK"
            ),
        }
    finally:
        db.close()

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
    """
    Voice search — listens from the microphone for <duration> seconds,
    transcribes using Vosk, then runs the normal text search.
    Returns the transcribed text alongside the results so the frontend
    can show "Searching for: dogs at the park".
    """
    try:
        transcribed = voice_engine.listen_and_transcribe(duration=duration)
        if not transcribed or not transcribed.strip():
            return {"status": "error", "message": "Could not hear anything. Speak clearly and try again."}
        logger.info(f"🎤 Voice transcribed: '{transcribed}'")
        result = search(query=transcribed.strip(), top_k=20)
        result["transcribed"] = transcribed.strip()
        return result
    except Exception as e:
        logger.error(f"Voice search error: {e}", exc_info=True)
        return {"status": "error", "message": f"Voice search failed: {str(e)}"}


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(20)):
    """
    Reverse image search — upload any photo and find visually similar ones.
    Uses the CLIP image encoder so results are semantically meaningful
    (a photo of a golden retriever finds other dog photos, not just identical shots).
    """
    if search_engine.index is None:
        return {"status": "error", "message": "No images indexed. Run build_index.py first."}

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPG, PNG or WebP.")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        query_emb = search_engine.get_image_embedding(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if query_emb is None:
        return {"status": "error", "message": "Could not process the uploaded image."}

    total = search_engine.index.ntotal
    q = query_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    distances, indices = search_engine.index.search(q, min(top_k * 3, total))

    db = SessionLocal()
    try:
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            score = float(dist)
            if score < 0.45:          # CLIP image-to-image scores are generally higher than text-to-image
                break
            img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
            if not img:
                continue
            results.append({
                "id":           img.id,
                "filename":     img.filename,
                "score":        round(score * 100, 2),
                "timestamp":    img.timestamp.isoformat() if img.timestamp else None,
                "location":     {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
                "person_count": img.person_count or 0,
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        if not results:
            return {"status": "not_found", "message": "No visually similar images found."}

        logger.info(f"🖼️  Image search: {len(results)} results (top={results[0]['score']:.1f})")
        return {"status": "found", "count": len(results), "results": results}
    finally:
        db.close()


@app.post("/search/color")
def search_by_color(color: str = Form(...), top_k: int = Form(20)):
    """
    Pure color search — finds images whose average RGB color is closest to
    the requested color name.  Works independently of CLIP so "red photos"
    actually returns red-dominant images, not just images tagged 'red'.
    """
    COLOR_TARGETS = {
        "red":    (220,  50,  50),
        "orange": (230, 120,  40),
        "yellow": (220, 210,  50),
        "green":  ( 50, 160,  50),
        "blue":   ( 50,  80, 220),
        "purple": (120,  50, 180),
        "pink":   (230, 130, 160),
        "white":  (240, 240, 240),
        "black":  ( 20,  20,  20),
        "gray":   (128, 128, 128),
        "grey":   (128, 128, 128),
        "brown":  (140,  90,  50),
    }
    color_key = color.strip().lower()
    # Exact match first, then substring (handles "dark red" → "red")
    if color_key not in COLOR_TARGETS:
        color_key = next((k for k in COLOR_TARGETS if k in color_key), None)
    if not color_key:
        return {
            "status": "error",
            "message": f"Unknown color '{color}'. Supported: {', '.join(COLOR_TARGETS.keys())}"
        }

    target = np.array(COLOR_TARGETS[color_key], dtype=np.float32)

    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(
            DBImage.avg_r != None,
            DBImage.avg_g != None,
            DBImage.avg_b != None,
        ).all()
        if not images:
            return {"status": "not_found", "message": "No color data found. Rebuild the index."}

        scored = []
        for img in images:
            img_rgb = np.array([img.avg_r or 0, img.avg_g or 0, img.avg_b or 0], dtype=np.float32)
            dist    = float(np.linalg.norm(img_rgb - target))
            # Max possible distance is sqrt(3) * 255 ≈ 441.7
            score   = max(0.0, 100.0 * (1.0 - dist / 441.7))
            scored.append((score, img))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Minimum 25% similarity — avoids returning completely unrelated colors
        scored = [(s, img) for s, img in scored if s >= 25.0][:top_k]

        if not scored:
            return {"status": "not_found", "message": f"No images close to color '{color}'."}

        results = [{
            "id":           img.id,
            "filename":     img.filename,
            "score":        round(s, 2),
            "timestamp":    img.timestamp.isoformat() if img.timestamp else None,
            "location":     {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
            "person_count": img.person_count or 0,
        } for s, img in scored]

        logger.info(f"🎨 Color search '{color}': {len(results)} results")
        return {"status": "found", "query": color, "count": len(results), "results": results}
    finally:
        db.close()

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
        cover = f"/images/{images[0].filename}" if images else None
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
            cover = f"/images/{album_images[0].filename}"
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
        skipped_no_image = 0
        for fr in face_records:
            try:
                emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
                if emb.shape[0] != 512:
                    continue
                # Only cluster faces that are actually linked to an image
                if not fr.image_id:
                    skipped_no_image += 1
                    continue
                embeddings.append(emb)
                valid_face_records.append(fr)
            except Exception as e:
                logger.warning(f"Bad embedding face {fr.id}: {e}")

        logger.info(f"👥 {len(embeddings)} valid face embeddings "
                    f"({skipped_no_image} skipped — no image_id)")

        people_count = 0
        if embeddings:
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
                # valid_face_records[i] is in sync with embeddings[i] — safe
                valid_face_records[i].person_id = person_map[label]
            db.commit()
            logger.info(f"✅ {people_count} people created from {len(embeddings)} faces")
        else:
            logger.warning("⚠️  No face embeddings with valid image links — run build_index.py first")

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

    