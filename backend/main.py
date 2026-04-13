# # # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.staticfiles import StaticFiles
# # # import os, uuid, shutil, numpy as np, faiss
# # # from datetime import datetime
# # # import logging
# # # from contextlib import asynccontextmanager
# # # import datetime as datetime_module
# # # import json as _json
# # # from enhanced_ocr_engine import ocr_engine
# # # from image_captioning_engine import captioning_engine
# # # from quality_emotion_aesthetic_engines import (
# # #     image_quality,
# # #     emotion_detection,
# # #     aesthetic_scoring
# # # )

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
# # # from fastapi.responses import FileResponse
# # # from sqlalchemy import text
# # # import re
# # # import json
# # # from api_endpoints import router as dl_router
# # # from voice_route import router as voice_router
# # # from features_router import router as feat_router, ensure_extra_columns   # ← only import here

# # # _BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
# # # IMAGE_DIR        = os.path.normpath(os.path.join(_BASE_DIR, "..", "data", "images"))
# # # FAISS_INDEX_PATH = os.path.normpath(os.path.join(_BASE_DIR, "..", "data", "index.faiss"))

# # # FACE_MATCH_THRESHOLD  = float(os.environ.get("FACE_MATCH_THRESHOLD",  0.75))
# # # FACE_MATCH_NEIGHBORS  = int(os.environ.get("FACE_MATCH_NEIGHBORS",    5))
# # # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # # RECLUSTER_ON_UPLOAD   = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # # RECLUSTER_BATCH_SIZE  = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # # CLIP_SCORE_MIN  = float(os.environ.get("CLIP_SCORE_MIN",  0.14))  # 0.14: catches football/soccer/cat/dog matches
# # # THRESHOLD_FLOOR = float(os.environ.get("THRESHOLD_FLOOR", 0.22))
# # # ADAPTIVE_RATIO  = float(os.environ.get("ADAPTIVE_RATIO",  0.92))
# # # FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.13))  # 0.13: matches CLIP threshold

# # # RECLUSTER_COUNTER_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # # recluster_last_triggered = None

# # # COLOR_SCORE_MAP = {
# # #     'red':    (1.0, 0,    0),    'blue':   (0,   0,    1.0),
# # #     'green':  (0,   1.0,  0),    'yellow': (1.0, 1.0,  0),
# # #     'orange': (1.0, 0.5,  0),    'purple': (0.5, 0,    0.5),
# # #     'pink':   (1.0, 0.75, 0.8),  'black':  (0,   0,    0),
# # #     'white':  (1,   1,    1),    'gray':   (0.5, 0.5,  0.5),
# # #     'brown':  (0.6, 0.4,  0.2),
# # # }


# # # # ── Emotion emojis → search terms (intercept before CLIP) ───────────────────
# # # EMOTION_EMOJI_MAP = {
# # #     "😊": "happy",  "😄": "happy",  "😁": "happy",  "🙂": "happy",  "😀": "happy",
# # #     "😃": "happy",  "🤗": "happy",  "😆": "happy",  "😂": "happy",  "🥰": "happy",
# # #     "😍": "happy",  "😎": "happy",  "🥳": "happy",  "😇": "happy",
# # #     "😢": "sad",    "😭": "sad",    "😔": "sad",    "😟": "sad",    "🥺": "sad",
# # #     "😞": "sad",    "😿": "sad",    "💔": "sad",
# # #     "😠": "angry",  "😡": "angry",  "🤬": "angry",  "😤": "angry",  "👿": "angry",
# # #     "😲": "surprised", "😮": "surprised", "🤯": "surprised", "😱": "surprised",
# # #     "🤢": "disgusted", "🤮": "disgusted", "😷": "disgusted",
# # #     "😨": "fearful",   "😰": "fearful",   "😱": "fearful",   "😧": "fearful",
# # #     "😐": "neutral",   "😑": "neutral",   "🙄": "neutral",   "😶": "neutral",
# # # }

# # # # Extended scene emoji map (supplements search_engine.py EMOJI_MAP)
# # # SCENE_EMOJI_MAP = {
# # #     "🌅": "sunset golden hour", "🌃": "city night",   "🏕️": "camping outdoor",
# # #     "🌴": "tropical palm tree", "⛰️": "mountain",     "🌊": "ocean waves beach",
# # #     "🎊": "party celebration",  "👪": "family group",  "🤳": "selfie portrait",
# # #     "🐕": "dog",                "🐈": "cat",            "🦁": "lion animal",
# # #     "🍽️": "food meal",          "☕": "coffee",         "🍰": "cake dessert",
# # #     "🏋️": "gym exercise",       "🧘": "yoga meditation","🏊": "swimming pool",
# # #     "🌸": "flowers spring",     "🍂": "autumn fall leaves",
# # #     "🎸": "music guitar",       "📸": "camera photography",
# # #     "⚽": "soccer football sport", "🏀": "basketball sport",
# # #     "🏈": "american football sport", "🎾": "tennis sport",
# # #     "🏏": "cricket sport bat", "🏒": "hockey sport",
# # #     "⚾": "baseball sport",    "🥊": "boxing sport",
# # # }

# # # # Sport/activity keyword expansions (applied in _clean_query)
# # # SPORT_SYNONYMS = {
# # #     "football": "football soccer sport player",
# # #     "soccer":   "soccer football sport player",
# # #     "cricket":  "cricket sport bat player",
# # #     "basketball": "basketball sport player",
# # #     "tennis":   "tennis sport player",
# # #     "swimming": "swimming pool sport",
# # #     "running":  "running jogging sport",
# # #     "cycling":  "cycling bicycle sport",
# # #     "gym":      "gym workout exercise",
# # #     "yoga":     "yoga meditation exercise",
# # # }

# # # # ── Text keyword → emotion mapping ─────────────────────────────────────────
# # # EMOTION_KEYWORDS = {
# # #     "happy":     ["happy", "happiness", "smiling", "smile", "laughing", "laugh",
# # #                   "joy", "joyful", "cheerful", "glad", "delighted", "pleased"],
# # #     "sad":       ["sad", "sadness", "crying", "cry", "unhappy", "upset",
# # #                   "depressed", "gloomy", "sorrow", "sorrowful", "tears"],
# # #     "angry":     ["angry", "anger", "mad", "furious", "rage", "annoyed", "irritated"],
# # #     "surprised": ["surprised", "surprise", "shocked", "shock", "amazed", "astonished"],
# # #     "fearful":   ["fearful", "fear", "scared", "afraid", "frightened", "terrified"],
# # #     "disgusted": ["disgusted", "disgust", "disgusting"],
# # #     "neutral":   ["neutral", "expressionless", "blank"],
# # # }

# # # def _extract_emotion_from_query(query: str):
# # #     """Return emotion name if query contains an emotion emoji or keyword, else None."""
# # #     # Check emojis first
# # #     for emoji, emo in EMOTION_EMOJI_MAP.items():
# # #         if emoji in query:
# # #             return emo
# # #     # Check text keywords
# # #     q_lower = query.lower()
# # #     for emo, keywords in EMOTION_KEYWORDS.items():
# # #         if any(kw in q_lower for kw in keywords):
# # #             return emo
# # #     return None

# # # def _expand_query_emojis(query: str) -> str:
# # #     """Replace scene emojis with text equivalents for better CLIP embedding."""
# # #     result = query
# # #     for emoji, text in SCENE_EMOJI_MAP.items():
# # #         result = result.replace(emoji, " " + text + " ")
# # #     return " ".join(result.split())


# # # def _live(db):
# # #     return db.query(DBImage).filter(
# # #         (DBImage.is_trashed == False) | (DBImage.is_trashed == None)
# # #     )


# # # def _img_url(filename: str) -> str:
# # #     if not filename:
# # #         return None
# # #     return os.path.basename(filename)


# # # def _user_tags(img) -> list:
# # #     """Safely decode the user_tags JSON column into a Python list."""
# # #     try:
# # #         v = getattr(img, 'user_tags', None)
# # #         return json.loads(v) if v and v not in ('[]', '') else []
# # #     except Exception:
# # #         return []


# # # def _clean_query(query: str) -> str:
# # #     import unicodedata, re as _re
# # #     processed = resolve_query(query)
# # #     result = []
# # #     for char in processed:
# # #         cp = ord(char)
# # #         if cp > 127:
# # #             cat = unicodedata.category(char)
# # #             if cat in ("So", "Sm", "Sk", "Mn"):
# # #                 try:
# # #                     name = unicodedata.name(char, "").lower()
# # #                     name = name.replace(" sign", "").replace(" symbol", "")
# # #                     name = _re.sub(r"\bwith\b.*", "", name).strip()
# # #                     result.append(" " + name + " ")
# # #                 except Exception:
# # #                     pass
# # #             else:
# # #                 result.append(char)
# # #         else:
# # #             result.append(char)
# # #     cleaned = _re.sub(r"\s+", " ", "".join(result)).strip()
# # #     return cleaned if cleaned else processed


# # # def _startup_fix_filenames():
# # #     db = SessionLocal()
# # #     try:
# # #         fixed = 0
# # #         rows = db.query(DBImage).filter(DBImage.filename.contains("/")).all()
# # #         for r in rows:
# # #             r.filename = os.path.basename(r.filename)
# # #             fixed += 1
# # #         if fixed:
# # #             db.commit()
# # #             logger.info(f"Startup fix: normalised {fixed} filenames in DB")
# # #     except Exception as e:
# # #         db.rollback()
# # #         logger.warning(f"Startup filename fix failed: {e}")
# # #     finally:
# # #         db.close()


# # # def _enrich_image(image_id: int, file_path: str):
# # #     db = SessionLocal()
# # #     try:
# # #         img_record = db.query(DBImage).filter(DBImage.id == image_id).first()
# # #         if not img_record:
# # #             return
# # #         try:
# # #             img_record.caption_short    = captioning_engine.generate_caption(file_path, max_length=20)
# # #             img_record.caption_detailed = captioning_engine.generate_caption(file_path, max_length=60)
# # #             vqa_subject = captioning_engine.answer_visual_question(file_path, "What is the main subject in this image?")
# # #             vqa_person  = ""
# # #             if img_record.person_count and img_record.person_count > 0:
# # #                 # Ask BLIP to describe the person — identity recognition
# # #                 # works best with very short questions
# # #                 vqa_person = captioning_engine.answer_visual_question(
# # #                     file_path, "who is this?"
# # #                 )
# # #                 # Discard generic answers immediately
# # #                 if vqa_person and vqa_person.lower().strip() in {
# # #                     "man", "woman", "person", "a man", "a woman", "a person",
# # #                     "boy", "girl", "child", "human", "unknown", "celebrity",
# # #                     "actor", "actress", "no", "yes", "none", "i don't know",
# # #                 }:
# # #                     vqa_person = ""
# # #             img_record.caption_vqa = _json.dumps({
# # #                 "subject": vqa_subject,
# # #                 "person":  vqa_person,
# # #             } if (vqa_subject or vqa_person) else {})
# # #             img_record.caption_timestamp = datetime_module.datetime.now()
# # #             logger.info(f"✅ Caption [{image_id}]: {img_record.caption_short}")
# # #         except Exception as e:
# # #             logger.warning(f"⚠️  Caption failed [{image_id}]: {e}")
# # #         try:
# # #             img_record.ocr_text_enhanced = ocr_engine.extract_text(file_path)
# # #             kw = ocr_engine.extract_text_with_confidence(file_path)
# # #             img_record.ocr_keywords   = _json.dumps([i["text"] for i in kw if i["confidence"] > 0.7])
# # #             img_record.ocr_confidence = float(np.mean([i["confidence"] for i in kw])) if kw else 0.0
# # #             logger.info(f"✅ OCR [{image_id}]")
# # #         except Exception as e:
# # #             logger.warning(f"⚠️  OCR failed [{image_id}]: {e}")
# # #         try:
# # #             q = image_quality.assess_overall_quality(file_path)
# # #             img_record.quality_score = q.get("overall", 0)
# # #             img_record.quality_level = q.get("quality_level", "Unknown")
# # #             img_record.sharpness     = q.get("sharpness", 0)
# # #             img_record.exposure      = q.get("exposure", 0)
# # #             img_record.contrast      = q.get("contrast", 0)
# # #             img_record.composition   = q.get("composition", 0)
# # #             logger.info(f"✅ Quality [{image_id}]: {img_record.quality_level}")
# # #         except Exception as e:
# # #             logger.warning(f"⚠️  Quality failed [{image_id}]: {e}")
# # #         try:
# # #             # ── GATE: only run emotion model if InsightFace detected real human faces ──
# # #             # InsightFace (ArcFace) is accurate at human faces — won't detect cat faces,
# # #             # car headlights, or turtle heads. OpenCV Haar cascade (used by the emotion
# # #             # engine) will detect ALL of these as "faces" and classify them wrongly.
# # #             # Using the faces table count ensures we only run on real human images.
# # #             _face_count_check = db.query(DBFace).filter(DBFace.image_id == image_id).count()
# # #             if _face_count_check == 0:
# # #                 # No human faces detected by InsightFace → skip emotion model entirely
# # #                 img_record.dominant_emotion   = "neutral"
# # #                 img_record.face_emotion_count = 0
# # #                 img_record.emotion_data       = "[]"
# # #                 logger.info(f"⏭️  Emotion [{image_id}]: skipped (no InsightFace faces)")
# # #             else:
# # #                 ed = emotion_detection.detect_emotions(file_path)
# # #                 # ── Confidence filter: only keep high-confidence detections ──────────
# # #                 # Low-confidence results (< 0.50) from Haar cascade on non-frontal or
# # #                 # partially visible faces produce random emotion labels. Filter them out.
# # #                 ed = [e for e in ed if e.get("confidence", 0) >= 0.50]
# # #                 img_record.face_emotion_count = len(ed)
# # #                 img_record.dominant_emotion   = ed[0]["emotion"] if ed else "neutral"
# # #                 img_record.emotion_data       = _json.dumps(ed)
# # #                 logger.info(f"✅ Emotion [{image_id}]: {img_record.dominant_emotion} ({len(ed)} faces)")
# # #         except Exception as e:
# # #             logger.warning(f"⚠️  Emotion failed [{image_id}]: {e}")
# # #         try:
# # #             a = aesthetic_scoring.score_aesthetics(file_path)
# # #             img_record.aesthetic_score  = a.get("aesthetic_score", 0)
# # #             img_record.aesthetic_rating = a.get("rating", "Unknown")
# # #             logger.info(f"✅ Aesthetic [{image_id}]: {img_record.aesthetic_rating}")
# # #         except Exception as e:
# # #             logger.warning(f"⚠️  Aesthetic failed [{image_id}]: {e}")
# # #         db.commit()
# # #         logger.info(f"✅ Enrichment done for image {image_id}")
# # #     except Exception as e:
# # #         db.rollback()
# # #         logger.error(f"❌ Enrichment failed [{image_id}]: {e}", exc_info=True)
# # #     finally:
# # #         db.close()


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
# # #             except Exception:
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
# # #             logger.info("📊 Recluster triggered")
# # #             background_tasks.add_task(recluster)
# # #             recluster_last_triggered = now
# # #             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
# # #                 f.write('0')
# # #     except Exception as e:
# # #         logger.warning(f"Recluster check failed: {e}")


# # # # ── Lifespan ──────────────────────────────────────────────────────────────────
# # # @asynccontextmanager
# # # async def lifespan(app: FastAPI):
# # #     init_db()
# # #     ensure_extra_columns()          # ← runs safely here, after DB is initialised
# # #     _startup_fix_filenames()
# # #     if os.path.exists(FAISS_INDEX_PATH):
# # #         try:
# # #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# # #             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
# # #         except Exception as e:
# # #             logger.error(f"Index load failed: {e}")
# # #             search_engine.index = None
# # #     try:
# # #         _db = SessionLocal()
# # #         try:
# # #             faces_total    = _db.query(DBFace).filter(DBFace.face_embedding != None).count()
# # #             faces_assigned = _db.query(DBFace).filter(DBFace.face_embedding != None, DBFace.person_id != None).count()
# # #             people_total  = _db.query(Person).count()
# # #             imgs_in_album = _db.query(DBImage).filter(DBImage.album_id != None).count()
# # #             total_imgs    = _db.query(DBImage).count()
# # #             needs_face_repair  = faces_total > 0 and (faces_assigned == 0 or people_total == 0)
# # #             needs_album_repair = total_imgs  > 0 and imgs_in_album == 0
# # #         finally:
# # #             _db.close()
# # #         if needs_face_repair or needs_album_repair:
# # #             logger.warning(f"⚠️  Stale data — faces={faces_total} assigned={faces_assigned} people={people_total} imgs_in_album={imgs_in_album}/{total_imgs}")
# # #             logger.info("🔄 Auto-repairing on startup...")
# # #             recluster()
# # #             logger.info("✅ Auto-repair done")
# # #         else:
# # #             logger.info(f"✅ DB healthy: {faces_assigned}/{faces_total} faces assigned, {people_total} people, {imgs_in_album}/{total_imgs} imgs in albums")
# # #     except Exception as e:
# # #         logger.warning(f"Auto-repair check failed (non-fatal): {e}")
# # #     logger.info("✅ Ready!")
# # #     yield


# # # # ── app = FastAPI() MUST be here, before any app.include_router() ─────────────
# # # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)

# # # app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# # # from starlette.middleware.base import BaseHTTPMiddleware
# # # from starlette.requests import Request as StarletteRequest

# # # class FixImagePathMiddleware(BaseHTTPMiddleware):
# # #     async def dispatch(self, request: StarletteRequest, call_next):
# # #         path = request.scope["path"]
# # #         if path.startswith("/image/"):
# # #             suffix = path[len("/image/"):]
# # #             suffix = suffix.lstrip("/")
# # #             if suffix.startswith("images/"):
# # #                 suffix = suffix[len("images/"):]
# # #             bare = os.path.basename(suffix)
# # #             if bare:
# # #                 request.scope["path"] = f"/image/{bare}"
# # #         return await call_next(request)

# # # app.add_middleware(FixImagePathMiddleware)

# # # if not os.path.exists(IMAGE_DIR):
# # #     os.makedirs(IMAGE_DIR)

# # # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # # # ── All routers registered AFTER app = FastAPI() ──────────────────────────────
# # # app.include_router(dl_router)
# # # app.include_router(voice_router)
# # # app.include_router(feat_router)    # ← safely here, app already exists


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # UTILITY
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.get("/health")
# # # def health():
# # #     return {"status": "ready", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# # # @app.get("/debug")
# # # def debug_db():
# # #     db = SessionLocal()
# # #     try:
# # #         total_images         = db.query(DBImage).count()
# # #         total_faces          = db.query(DBFace).count()
# # #         faces_with_embedding = db.query(DBFace).filter(DBFace.face_embedding != None).count()
# # #         faces_with_image_id  = db.query(DBFace).filter(DBFace.image_id != None).count()
# # #         faces_with_person    = db.query(DBFace).filter(DBFace.person_id != None).count()
# # #         total_people         = db.query(Person).count()
# # #         total_albums         = db.query(Album).count()
# # #         images_in_album      = db.query(DBImage).filter(DBImage.album_id != None).count()
# # #         people_detail = []
# # #         for p in db.query(Person).all():
# # #             faces   = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# # #             img_ids = [f.image_id for f in faces if f.image_id]
# # #             people_detail.append({"id": p.id, "name": p.name, "face_count": len(faces), "faces_with_image_id": len(img_ids)})
# # #         album_detail = []
# # #         for a in db.query(Album).all():
# # #             imgs = db.query(DBImage).filter(DBImage.album_id == a.id).count()
# # #             album_detail.append({"id": a.id, "title": a.title, "image_count": imgs})
# # #         return {
# # #             "images": total_images, "faces": total_faces,
# # #             "faces_with_embedding": faces_with_embedding, "faces_with_image_id": faces_with_image_id,
# # #             "faces_with_person": faces_with_person, "people": total_people,
# # #             "albums": total_albums, "images_in_album": images_in_album,
# # #             "people_detail": people_detail, "album_detail": album_detail,
# # #             "action_needed": "Run POST /recluster to fix person/album assignments" if faces_with_person == 0 or images_in_album == 0 else "OK",
# # #         }
# # #     finally:
# # #         db.close()

# # # @app.get("/test-db")
# # # def test_db():
# # #     db = SessionLocal()
# # #     try:
# # #         count  = db.query(DBImage).count()
# # #         images = db.query(DBImage).limit(1).all()
# # #         return {
# # #             "status": "ok", "total_images": count,
# # #             "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None,
# # #         }
# # #     except Exception as e:
# # #         return {"status": "error", "message": str(e)}
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # IMAGE LISTING & SERVING
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.get("/images")
# # # def get_all_images(limit: int = 100):
# # #     db = SessionLocal()
# # #     try:
# # #         images = _live(db).limit(limit).all()
# # #         return {"results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "face_emotion_count": img.face_emotion_count, "aesthetic_score": img.aesthetic_score, "aesthetic_rating": img.aesthetic_rating, "caption_short": img.caption_short, "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
# # #     finally:
# # #         db.close()

# # # @app.get("/image/{filename:path}")
# # # def get_image_file(filename: str):
# # #     bare = os.path.basename(filename.lstrip("/"))
# # #     if not bare or bare in (".", ".."):
# # #         raise HTTPException(status_code=400, detail="Invalid filename")
# # #     candidates = [
# # #         os.path.join(IMAGE_DIR, bare),
# # #         os.path.normpath(os.path.join(os.getcwd(), "..", "data", "images", bare)),
# # #         os.path.normpath(os.path.join(os.getcwd(), "data", "images", bare)),
# # #         os.path.normpath(os.path.join(os.getcwd(), "images", bare)),
# # #     ]
# # #     for p in candidates:
# # #         if os.path.exists(p):
# # #             return FileResponse(p)
# # #     db = SessionLocal()
# # #     try:
# # #         img = db.query(DBImage).filter((DBImage.filename == bare) | (DBImage.filename == f"/images/{bare}")).first()
# # #         if img and img.original_path:
# # #             p = os.path.normpath(img.original_path.replace("//", "/"))
# # #             if os.path.exists(p):
# # #                 return FileResponse(p)
# # #             alt = os.path.join(IMAGE_DIR, os.path.basename(img.original_path))
# # #             if os.path.exists(alt):
# # #                 return FileResponse(alt)
# # #     finally:
# # #         db.close()
# # #     logger.warning(f"404 image bare={bare!r} IMAGE_DIR={IMAGE_DIR}")
# # #     raise HTTPException(status_code=404, detail=f"Image not found: {bare}")

# # # @app.get("/timeline")
# # # def get_timeline():
# # #     db = SessionLocal()
# # #     try:
# # #         images = _live(db).order_by(DBImage.timestamp.desc()).all()
# # #         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "face_emotion_count": img.face_emotion_count, "aesthetic_score": img.aesthetic_score, "aesthetic_rating": img.aesthetic_rating, "caption_short": img.caption_short, "ocr_text_enhanced": img.ocr_text_enhanced, "person_count": img.person_count, "width": img.width, "height": img.height, "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # DELETE / TRASH / RESTORE
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.post("/delete_image")
# # # def delete_image(image_id: int = Form(...)):
# # #     db = SessionLocal()
# # #     try:
# # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # #         if not img:
# # #             raise HTTPException(status_code=404, detail="Image not found")
# # #         img.is_trashed = True
# # #         img.trashed_at = datetime_module.datetime.now()
# # #         db.commit()
# # #         return {"success": True, "message": "Image moved to trash"}
# # #     finally:
# # #         db.close()

# # # @app.post("/restore")
# # # def restore_image(image_id: int = Form(...)):
# # #     db = SessionLocal()
# # #     try:
# # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # #         if not img:
# # #             raise HTTPException(status_code=404, detail="Image not found")
# # #         img.is_trashed = False
# # #         img.trashed_at = None
# # #         db.commit()
# # #         return {"success": True, "message": "Image restored"}
# # #     finally:
# # #         db.close()

# # # @app.post("/permanent_delete")
# # # def permanent_delete(image_id: int = Form(...)):
# # #     db = SessionLocal()
# # #     try:
# # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # #         if not img:
# # #             raise HTTPException(status_code=404, detail="Image not found")
# # #         filename  = img.filename
# # #         file_path = img.original_path or os.path.join(IMAGE_DIR, filename)
# # #         bare = os.path.basename(file_path)
# # #         for p in [file_path, os.path.join(IMAGE_DIR, bare)]:
# # #             if p and os.path.exists(p):
# # #                 try:
# # #                     os.remove(p)
# # #                     break
# # #                 except Exception as e:
# # #                     logger.warning(f"Could not delete file: {e}")
# # #         if search_engine.index is not None:
# # #             try:
# # #                 # Fast: remove just this one vector — no need to re-embed everything
# # #                 ids_arr = np.array([image_id], dtype="int64")
# # #                 search_engine.index.remove_ids(ids_arr)
# # #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# # #                 logger.info(f"✅ FAISS vector {image_id} removed")
# # #             except Exception as e:
# # #                 logger.warning(f"FAISS remove (non-critical): {e}")
# # #         db.query(DBFace).filter(DBFace.image_id == image_id).delete()
# # #         db.delete(img)
# # #         db.commit()
# # #         return {"success": True, "message": f"Image '{filename}' permanently deleted"}
# # #     except Exception as e:
# # #         db.rollback()
# # #         raise HTTPException(status_code=500, detail=str(e))
# # #     finally:
# # #         db.close()

# # # @app.get("/trash")
# # # def get_trash():
# # #     db = SessionLocal()
# # #     try:
# # #         images = db.query(DBImage).filter(DBImage.is_trashed == True).all()
# # #         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "trashed_at": img.trashed_at.isoformat() if img.trashed_at else None, "timestamp": img.timestamp.isoformat() if img.timestamp else None, "caption": img.caption_short or ""} for img in images]}
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # FAVORITES
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.post("/toggle_favorite")
# # # def toggle_favorite(image_id: int = Form(...)):
# # #     db = SessionLocal()
# # #     try:
# # #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# # #         if not img:
# # #             raise HTTPException(status_code=404, detail="Image not found")
# # #         img.is_favorite = not img.is_favorite
# # #         db.commit()
# # #         return {"success": True, "is_favorite": img.is_favorite}
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
# # #         images = _live(db).filter(DBImage.is_favorite == True).all()
# # #         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # SEARCH
# # # # ────────────────────────────────────────────────────────────────────────────
# # # def _score_candidates(query_emb, sig_words, query_colors, db, top_k, extra_emb=None, text_weight=1.0, image_weight=0.0):
# # #     if search_engine.index is None:
# # #         return []
# # #     if extra_emb is not None:
# # #         blended = text_weight * query_emb + image_weight * extra_emb
# # #         norm = np.linalg.norm(blended)
# # #         if norm > 1e-8:
# # #             blended /= norm
# # #         q_vec = blended
# # #     else:
# # #         q_vec = query_emb
# # #     total = search_engine.index.ntotal
# # #     q = q_vec.reshape(1, -1).astype("float32")
# # #     faiss.normalize_L2(q)
# # #     distances, indices = search_engine.index.search(q, total)
# # #     all_candidates = []
# # #     for dist, idx in zip(distances[0], indices[0]):
# # #         if idx == -1:
# # #             continue
# # #         clip_score = float(dist)
# # #         if clip_score < CLIP_SCORE_MIN:
# # #             break
# # #         img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# # #         if not img or img.is_trashed:
# # #             continue
# # #         # (empty-caption guard removed — was killing valid results for colour queries)
# # #         # ── OCR bonus (only if most sig_words appear) ──────────────────────
# # #         ocr_bonus = 0.0
# # #         ocr_src = (img.ocr_text_enhanced or "") + " " + (getattr(img, "ocr_keywords", "") or "")
# # #         if sig_words and ocr_src.strip():
# # #             txt  = ocr_src.lower()
# # #             hits = sum(1 for w in sig_words if w in txt)
# # #             # Require at least half the words to match to avoid false boosts
# # #             if hits >= max(1, len(sig_words) * 0.5):
# # #                 ocr_bonus = min(hits / len(sig_words), 1.0)

# # #         # ── Tag bonus ──────────────────────────────────────────────────────
# # #         tag_bonus = 0.0
# # #         all_tags_text = " ".join(_user_tags(img))
# # #         if img.scene_label:
# # #             all_tags_text += " " + img.scene_label.lower()
# # #         if all_tags_text.strip() and any(w in all_tags_text for w in sig_words):
# # #             tag_bonus = 1.0

# # #         # ── Caption bonus + contradiction filter ─────────────────────────
# # #         caption_bonus = 0.0
# # #         hard_exclude    = False  # True = skip this result entirely
# # #         caption_bonus   = 0.0
# # #         caption_penalty = 0.0
# # #         caption_src = " ".join(filter(None, [
# # #             img.caption_short or "", img.caption_detailed or ""
# # #         ])).lower()

# # #         # ── Shared sets (always defined — needed even when sig_words=[]) ──
# # #         def _stem(w):
# # #             if len(w) > 4 and w.endswith("ses"): return w[:-2]
# # #             if len(w) > 3 and w.endswith("ies"): return w[:-3]+"y"
# # #             if len(w) > 3 and w.endswith("es"):  return w[:-2]
# # #             if len(w) > 3 and w.endswith("s"):   return w[:-1]
# # #             return w

# # #         _COLOURS = {"black","white","red","blue","green","yellow","orange",
# # #                     "purple","pink","brown","gray","grey","golden","silver",
# # #                     "nude","beige","cream","navy","teal","maroon","violet"}
# # #         _ANIMALS = {
# # #             "horse","cow","dog","cat","bird","elephant","tiger","lion",
# # #             "bear","wolf","deer","sheep","goat","pig","rabbit","snake",
# # #             "fish","chicken","duck","frog","butterfly","spider",
# # #             "fox","monkey","giraffe","zebra","panda","kangaroo","crocodile",
# # #             "alligator","rhino","hippo","whale","dolphin","shark","eagle",
# # #             "parrot","penguin","owl","bat","beetle","ant","bee","crab",
# # #             "donkey","camel","llama","raccoon","squirrel","hamster",
# # #             "mouse","rat","gecko","lizard","tortoise","turtle","swan",
# # #             "otter","seal","walrus","beaver","meerkat","flamingo","peacock",
# # #             "cheetah","leopard","jaguar","gorilla","chimpanzee","baboon",
# # #             "koala","wombat","hedgehog","ferret","chinchilla","iguana",
# # #             "chameleon","scorpion","jellyfish","octopus","lobster","shrimp",
# # #             "kitten","puppy","calf","foal","piglet","lamb","cub","chick",
# # #         }
# # #         _CLOTHING = {"dress","suit","shirt","jacket","sari","skirt","gown",
# # #                      "uniform","coat","blouse","top","hoodie","sweater",
# # #                      "saree","kurta","lehenga","tuxedo","blazer","vest",
# # #                      "bikini","swimsuit","robe","kimono","pyjama","shorts",
# # #                      "trouser","pant","jeans","legging"}
# # #         _GENDER = {"woman","girl","lady","female","man","boy","male","guy"}
# # #         _FEMALE = {"woman","girl","lady","female"}
# # #         _MALE   = {"man","boy","guy","male"}
# # #         _HUMAN_WORDS = {"man","woman","boy","girl","person","people","lady",
# # #                         "guy","child","adult","player","actor","actress"}

# # #         # Always compute q_* from sig_words (empty set when sig_words=[])
# # #         _q_set     = set(sig_words)
# # #         q_colours  = _COLOURS  & _q_set
# # #         q_animals  = _ANIMALS  & _q_set
# # #         q_clothing = _CLOTHING & _q_set
# # #         q_gender   = _GENDER   & _q_set

# # #         # Always compute cap_words from caption
# # #         if caption_src:
# # #             _cap_raw  = caption_src.split()
# # #             cap_words = {_stem(w) for w in _cap_raw} | set(_cap_raw)
# # #         else:
# # #             cap_words = set()

# # #         if sig_words and caption_src:
# # #             # ── Caption bonus ────────────────────────────────────────────
# # #             hits = sum(1 for w in _q_set if w in caption_src)
# # #             match_ratio = hits / len(_q_set)
# # #             if match_ratio >= 0.5:
# # #                 caption_bonus = match_ratio

# # #             # ── 1. Animal mismatch ───────────────────────────────────────
# # #             if q_animals:
# # #                 cap_animals = _ANIMALS & cap_words
# # #                 if cap_animals and not (cap_animals & q_animals):
# # #                     hard_exclude = True

# # #             # ── 2. Colour + clothing contradiction (context-aware) ──────
# # #             if not hard_exclude and q_colours:
# # #                 cap_colours  = _COLOURS & cap_words
# # #                 cap_clothing = _CLOTHING & cap_words
# # #                 same_cloth   = bool(cap_clothing & q_clothing)
# # #                 contradicting = cap_colours - q_colours
# # #                 matching      = cap_colours & q_colours
# # #                 if same_cloth and cap_colours:
# # #                     # Caption has same clothing item → colour must match
# # #                     # e.g. "white dress" for "black dress" → EXCLUDED
# # #                     # "black hair wearing white dress" → "black" matches query but 
# # #                     # the DRESS colour (white) contradicts → still exclude
# # #                     if contradicting and not matching:
# # #                         hard_exclude = True
# # #                     elif contradicting and matching:
# # #                         caption_penalty += 0.10
# # #                 elif not same_cloth and cap_colours and contradicting and not matching:
# # #                     caption_penalty += 0.18

# # #                         # ── 3. Clothing constraint — hard exclude on mismatch ────────
# # #             if not hard_exclude and q_clothing:
# # #                 cap_clothing = _CLOTHING & cap_words
# # #                 if cap_clothing and not (cap_clothing & q_clothing):
# # #                     # Caption names a DIFFERENT clothing item (e.g. "suit" for "dress")
# # #                     hard_exclude = True
# # #                 elif not cap_clothing:
# # #                     # Caption has NO clothing word at all
# # #                     if q_colours and (q_colours & cap_words):
# # #                         # Colour word appears in caption (e.g. "red eyes") but no clothing
# # #                         # → the colour belongs to something else, not the dress → exclude
# # #                         hard_exclude = True
# # #                     elif q_colours and not (q_colours & cap_words):
# # #                         # No clothing AND wrong/no colour → definitely not the right image
# # #                         hard_exclude = True

# # #             # ── 4. Gender mismatch ───────────────────────────────────────
# # #             if not hard_exclude and q_gender:
# # #                 q_female = bool(_FEMALE & q_gender)
# # #                 q_male   = bool(_MALE   & q_gender)
# # #                 if q_female and (_MALE & cap_words) and not (_FEMALE & cap_words):
# # #                     hard_exclude = True   # query=woman but caption has only men → always exclude
# # #                 if q_male and (_FEMALE & cap_words) and not (_MALE & cap_words):
# # #                     hard_exclude = True   # query=man but caption has only women → always exclude

# # #             if caption_penalty >= 0.40:
# # #                 hard_exclude = True
# # #             caption_penalty = min(caption_penalty, 0.38)

# # #         # ── Animal query: exclude human-only captions (always runs) ──────
# # #         if not hard_exclude and q_animals and not q_gender and caption_src:
# # #             cap_has_human  = bool(_HUMAN_WORDS & cap_words)
# # #             cap_has_animal = bool(_ANIMALS & cap_words)
# # #             if cap_has_human and not cap_has_animal:
# # #                 hard_exclude = True

# # #         # ── BUGFIX: Human query: exclude animal-only captions ───────────
# # #         if not hard_exclude and (_HUMAN_WORDS & _q_set) and caption_src:
# # #             cap_has_human  = bool(_HUMAN_WORDS & cap_words)
# # #             cap_has_animal = bool(_ANIMALS & cap_words)
# # #             if cap_has_animal and not cap_has_human:
# # #                 hard_exclude = True

# # #         # ── BUGFIX: Empty/missing caption — use scene_label as fallback ─
# # #         # When caption_src is empty (BLIP not yet run or failed), ALL the
# # #         # animal/human filters above are skipped because they check
# # #         # `if sig_words and caption_src`.  Non-matching animals then leak
# # #         # through with a pure CLIP score.  Use scene_label + user_tags to
# # #         # catch the worst offenders.
# # #         if not hard_exclude and not caption_src and q_animals:
# # #             # Build a stem-aware token set from scene_label + user_tags
# # #             _sl = (img.scene_label or "").lower()
# # #             _ut = " ".join(_user_tags(img)).lower()
# # #             _fb_raw = set((_sl + " " + _ut).split())
# # #             _fb_toks = _fb_raw | {_stem(w) for w in _fb_raw}
# # #             _fb_animals = _ANIMALS & _fb_toks
# # #             # If scene_label names a DIFFERENT animal from the query → exclude
# # #             if _fb_animals and not (_fb_animals & q_animals):
# # #                 hard_exclude = True
# # #             # If scene_label has NO animal at all but has human words → exclude
# # #             # (e.g. "person, sink" scene_label for a cat query)
# # #             if not _fb_animals:
# # #                 _fb_humans = _HUMAN_WORDS & _fb_toks
# # #                 if _fb_humans:
# # #                     hard_exclude = True

# # #         # ── BUGFIX: Specific object words absent from caption ────────────────
# # #         # If query mentions a concrete specific object (flower, rose, guitar, etc.)
# # #         # and the caption exists but doesn't mention it at all → CLIP is hallucinating
# # #         # semantic similarity. These concrete words almost never produce false negatives
# # #         # (if the image has a flower, BLIP will say "flower").
# # #         _SPECIFIC_OBJECTS = {
# # #             "flower","rose","tulip","sunflower","daisy","bouquet","petal",
# # #             "guitar","piano","violin","drum","trumpet","saxophone",
# # #             "cake","pizza","burger","sandwich","coffee","wine","beer","cocktail",
# # #             "trophy","medal","cup","award","crown",
# # #             "balloon","umbrella","candle","lantern","flag",
# # #             "sword","shield","arrow","bow",
# # #             "basketball","football","soccer","tennis","cricket","baseball",
# # #             "bicycle","motorcycle","skateboard","surfboard","snowboard",
# # #         }
# # #         if not hard_exclude and caption_src and sig_words:
# # #             _q_specific = _SPECIFIC_OBJECTS & set(sig_words)
# # #             if _q_specific:
# # #                 # Any of the specific objects must appear in caption (stemmed)
# # #                 _q_specific_stemmed = {_stem(w) for w in _q_specific} | _q_specific
# # #                 if not (_q_specific_stemmed & cap_words):
# # #                     hard_exclude = True

# # #         # ── Hard exclude — skip entirely ────────────────────────────────
# # #         if hard_exclude:
# # #             continue

# # #         # ── Color bonus ──────────────────────────────────────────────────
# # #         color_bonus = 0.0
# # #         if query_colors and getattr(img, "avg_r", None) is not None:
# # #             img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# # #             for qc in query_colors:
# # #                 d = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# # #                 color_bonus = max(color_bonus, max(0.0, 1.0 - d / np.sqrt(3)))

# # #         # ── Final score ──────────────────────────────────────────────────
# # #         # Strong "all keywords in caption" bonus — gives confirmed matches a
# # #         # decisive edge over generic CLIP matches with the same cosine score.
# # #         exact_match_bonus = 0.0
# # #         if sig_words and caption_src:
# # #             all_present = all(w in caption_src for w in sig_words)
# # #             majority_present = sum(1 for w in sig_words if w in caption_src) / len(sig_words) >= 0.7
# # #             if all_present:
# # #                 exact_match_bonus = 0.25   # ALL query words in caption → guaranteed top
# # #             elif majority_present:
# # #                 exact_match_bonus = 0.12   # Most query words → still boosted
# # #         final_score = (0.60 * clip_score          # CLIP: directional signal
# # #                       + 0.20 * caption_bonus       # caption match boost
# # #                       + 0.15 * exact_match_bonus   # EXACT match → decisive top rank
# # #                       + 0.03 * tag_bonus
# # #                       + 0.01 * ocr_bonus
# # #                       + 0.01 * color_bonus
# # #                       - caption_penalty)
# # #         all_candidates.append({"img": img, "clip": clip_score, "final": final_score})
# # #     all_candidates.sort(key=lambda x: x["final"], reverse=True)
# # #     kept = [c for c in all_candidates if c["final"] >= FINAL_SCORE_MIN]
# # #     if all_candidates:
# # #         logger.info(f"📊 CLIP: {len(all_candidates)} above CLIP_MIN, {len(kept)} above FINAL_MIN={FINAL_SCORE_MIN}, top={all_candidates[0]['final']:.3f} bottom={all_candidates[-1]['final']:.3f}")
# # #     results = []
# # #     for c in kept[:top_k]:
# # #         img = c["img"]
# # #         results.append({
# # #                 "id": img.id,
# # #                 "filename": _img_url(img.filename),
# # #                 "score": round(c["final"] * 100, 2),
# # #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # #                 "person_count": img.person_count or 0,
# # #                 "caption_short": img.caption_short or "",
# # #                 "caption_detailed": img.caption_detailed or "",
# # #                 "quality_score": img.quality_score or 0,
# # #                 "quality_level": img.quality_level or "",
# # #                 "sharpness": img.sharpness or 0,
# # #                 "exposure": img.exposure or 0,
# # #                 "contrast": img.contrast or 0,
# # #                 "composition": img.composition or 0,
# # #                 "dominant_emotion": img.dominant_emotion or "",
# # #                 "face_emotion_count": img.face_emotion_count or 0,
# # #                 "aesthetic_score": img.aesthetic_score or 0,
# # #                 "aesthetic_rating": img.aesthetic_rating or "",
# # #                 "ocr_text_enhanced": img.ocr_text_enhanced or "",
# # #                 "scene_label": img.scene_label or "",
# # #                 "width": img.width,
# # #                 "height": img.height,
# # #                 "is_favorite": bool(img.is_favorite),
# # #                 "user_tags": _user_tags(img),
# # #                 "photo_note": getattr(img, "photo_note", "") or "",
# # #             })
# # #     return results

# # # @app.post("/search")
# # # def search(query: str = Form(...), top_k: int = Form(20)):
# # #     if not query or not query.strip():
# # #         return {"status": "error", "message": "Query empty"}

# # #     # ── Emotion search: emoji OR text keywords → filter by dominant_emotion ──
# # #     detected_emotion = _extract_emotion_from_query(query)
# # #     is_pure_emoji    = query.strip() in EMOTION_EMOJI_MAP

# # #     # Check if query is MOSTLY about emotion (emotion word + optional "faces"/"photos" etc)
# # #     EMOTION_MODIFIERS = {"faces", "face", "photos", "photo", "pictures", "picture",
# # #                          "images", "image", "people", "person", "moments", "looking",
# # #                          "expression", "expressions", "ones", "me", "us", "all",
# # #                          # BUGFIX: common query prefixes that were left in non_emotion_words
# # #                          # causing "show me happy faces" / "find sad photos" to miss the
# # #                          # emotion route and fall through to CLIP which returns nothing.
# # #                          "show", "find", "get", "give", "display", "share",
# # #                          "my", "the", "a", "with", "some", "any"}
# # #     query_words = set(query.lower().split())
# # #     non_emotion_words = query_words - EMOTION_MODIFIERS -         {kw for kws in EMOTION_KEYWORDS.values() for kw in kws}
# # #     is_emotion_query = detected_emotion and len(non_emotion_words) == 0

# # #     if detected_emotion and (is_pure_emoji or is_emotion_query):
# # #         # Route directly to emotion DB filter
# # #         # Only match images where emotion was detected by face analysis
# # #         # (face_emotion_count > 0 means actual faces were detected and analysed)
# # #         db = SessionLocal()
# # #         try:
# # #             # Require: actual faces detected (face_emotion_count>0) AND human present (person_count>0)
# # #             # This prevents turtles/cars/cats from appearing in happy/angry emotion search
# # #             # Strict: only images where faces were actually detected + analysed
# # #             # face_emotion_count > 0 = real face detection ran on this image
# # #             imgs = _live(db).filter(
# # #                 DBImage.dominant_emotion == detected_emotion,
# # #                 DBImage.face_emotion_count > 0,
# # #             ).order_by(DBImage.timestamp.desc()).limit(top_k).all()
# # #             # NO fallback that removes face_emotion_count requirement.
# # #             # Without this guard, cars/turtles/spider-man with wrongly assigned
# # #             # emotion labels appear in results. Better to show fewer real results.
# # #             if imgs:
# # #                 return {"status": "found", "query": query, "count": len(imgs),
# # #                         "results": [{"id": img.id, "filename": _img_url(img.filename),
# # #                                      "score": 90.0,
# # #                                      "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                                      "caption_short": img.caption_short or "",
# # #                                      "user_tags": _user_tags(img),
# # #                                      "dominant_emotion": img.dominant_emotion,
# # #                                      "person_count": img.person_count or 0,
# # #                                      "photo_note": getattr(img,"photo_note","") or ""} for img in imgs]}
# # #             else:
# # #                 # No results — emotions may not be detected yet
# # #                 return {"status": "not_found",
# # #                         "message": f"No photos with '{detected_emotion}' emotion found. "
# # #                                    f"Click 'Fix Emotions' in the sidebar to detect emotions on your photos."}
# # #         finally:
# # #             db.close()

# # #     # ── Expand scene emojis to text for CLIP ────────────────────────────────
# # #     expanded_query = _expand_query_emojis(query)
# # #     # Expand sport/activity synonyms for better CLIP recall
# # #     _words = expanded_query.lower().split()
# # #     if any(w in SPORT_SYNONYMS for w in _words):
# # #         _words_exp = []
# # #         for w in _words:
# # #             _words_exp.append(SPORT_SYNONYMS.get(w, w))
# # #         expanded_query = " ".join(_words_exp)
# # #     processed_query = _clean_query(expanded_query)
# # #     logger.info(f"🔍 Search: '{query}' → '{processed_query}'")

# # #     # BUGFIX: q_tokens_pre must be defined HERE — it is used by the
# # #     # should_search_people pre-computation block below, which runs before
# # #     # the later assignment at the CLIP section (line ~17609).
# # #     q_tokens_pre = query.lower().strip().split()

# # #     # If emotion emoji mixed with other words, boost emotion filter
# # #     emotion_boost = detected_emotion  # may be None

# # #     # ── Pre-compute should_search_people so sig_words can use it ────────────
# # #     # (Full person-search logic runs later; here we just need the flag)
# # #     _DESCRIPTOR_PRE = {
# # #         "a","an","the","in","on","at","of","for","with","by","to","from","into",
# # #         "is","are","was","were","be","been","has","have","had","do","does","did",
# # #         "will","would","could","should","may","might","can","wearing","holding",
# # #         "standing","sitting","posing","looking","long","short","tall","small",
# # #         "big","large","little","young","old","beautiful","pretty","black","white",
# # #         "red","blue","green","yellow","orange","purple","pink","brown","gray",
# # #         "woman","man","girl","boy","lady","guy","person","people","female","male",
# # #         "human","child","adult","baby","player","actor","photo","image","picture",
# # #         "standing","next","front","back","window","field","camera","dress","suit",
# # #         "shirt","jacket","hair","smiling",
# # #         # FIX: query-prefix words that aren't person names
# # #         "show","find","get","give","display","search","look","me","my","us","our",
# # #         # FIX: emotion/state adjectives that aren't person names
# # #         "happy","sad","angry","surprised","fearful","neutral","disgusted",
# # #         "smiling","laughing","crying","excited","bored","tired","scared",
# # #         # FIX: common descriptive adjectives that confused the name detector
# # #         "funny","cute","cool","nice","great","good","bad","best","worst",
# # #         "images","photos","pictures","moments","all","some","any","many",
# # #     }
# # #     # Animal/object words that look like names but are NOT people
# # #     _NON_PERSON_WORDS = {
# # #         # animals
# # #         "cat","dog","horse","cow","bird","fish","fox","lion","tiger","bear","wolf",
# # #         "pig","rabbit","duck","frog","snake","deer","sheep","goat","chicken","otter",
# # #         "seal","panda","koala","monkey","giraffe","zebra","elephant","whale","shark",
# # #         "eagle","owl","penguin","parrot","bee","ant","spider","butterfly","kitten",
# # #         "puppy",
# # #         # nature / places
# # #         "sunset","beach","ocean","mountain","forest","river","flower","tree",
# # #         "sky","snow","rain","night","city","park","road","bridge","building","house",
# # #         # vehicles / objects
# # #         "car","bus","train","plane","boat","food","pizza","cake","coffee",
# # #         # fictional characters & universes — prevent person-search triggering
# # #         "ironman","spiderman","batman","superman","thor","hulk","avengers",
# # #         "marvel","anime","cartoon","fictional","hero","villain","superhero",
# # #         "comic","spider","thanos","loki","deadpool","wolverine","aquaman",
# # #         "flash","wonder","hawkeye","ultron","venom","groot","rocket","nebula",
# # #         "gamora","starlord","antman","blackwidow","scarlet","vision","warmachine",
# # #         "falcon","panther","shuri","okoye","wakanda","asgard","xmen","cyclops",
# # #         "magneto","mystique","beast","nightcrawler","storm","phoenix","rogue",
# # #         # music genres / scene words (not person names)
# # #         "kpop","kdrama","jpop","bollywood","hollywood","netflix","disney","pixar",
# # #         "hiphop","hiphop","jazz","rock","pop","metal","indie","classical",
# # #         # common OCR / UI words that appear as image text
# # #         "user","login","password","email","search","home","menu","settings",
# # #         "button","click","submit","cancel","back","next","done","ok","yes","no",
# # #         "error","loading","please","enter","select","upload","download","share",
# # #         "follow","like","comment","post","profile","account","app","website",
# # #         # generic concept words
# # #         "vintage","retro","modern","classic","dark","light","minimal","abstract",
# # #         "nature","urban","street","indoor","outdoor","close","wide","macro",
# # #     }
# # #     _min_len_pre      = 3 if len(q_tokens_pre) <= 3 else 4
# # #     _nc_pre           = [w for w in q_tokens_pre if len(w) >= _min_len_pre
# # #                          and w not in _DESCRIPTOR_PRE
# # #                          and w not in _NON_PERSON_WORDS
# # #                          and w.isalpha()]
# # #     _dc_pre           = sum(1 for w in q_tokens_pre if w in _DESCRIPTOR_PRE)
# # #     _is_desc_pre      = _dc_pre >= len(q_tokens_pre) * 0.5
# # #     should_search_people = bool(_nc_pre) and not _is_desc_pre


# # #     # ── Tag fast-path ─────────────────────────────────────────────────────────
# # #     # BUGFIX: Arbitrary user tags (e.g. "birthday", "vacation", "grandma") only
# # #     # contributed a 0.03 weight bonus to CLIP scores, making them essentially
# # #     # invisible when typed in the search bar. Now: if ANY word in the query
# # #     # exactly matches a known user tag in the DB, we return those tagged images
# # #     # immediately (score=97) merged with any further CLIP/OD results.
# # #     # This runs BEFORE the OD and CLIP paths so tags always take priority.
# # #     _q_words_lower   = set(query.lower().strip().split())
# # #     _tag_fast_results = []
# # #     _tag_fast_ids     = set()
# # #     try:
# # #         _db_tag = SessionLocal()
# # #         try:
# # #             # Collect all distinct tags that exist in the DB
# # #             _all_tag_rows = _db_tag.execute(text(
# # #                 "SELECT DISTINCT user_tags FROM images "
# # #                 "WHERE user_tags IS NOT NULL AND user_tags != '[]' "
# # #                 "AND (is_trashed IS NULL OR is_trashed=0)"
# # #             )).fetchall()
# # #             import json as _json_tag
# # #             _known_tags = set()
# # #             for _trow in _all_tag_rows:
# # #                 try:
# # #                     for _t in _json_tag.loads(_trow[0] or "[]"):
# # #                         _known_tags.add(_t.strip().lower())
# # #                 except Exception:
# # #                     pass
# # #             # Find query words that are known tags
# # #             _matched_tags = _q_words_lower & _known_tags
# # #             if _matched_tags:
# # #                 for _img in _live(_db_tag).all():
# # #                     _itags = {t.strip().lower() for t in _user_tags(_img)}
# # #                     if _itags & _matched_tags:
# # #                         if _img.id not in _tag_fast_ids:
# # #                             _tag_fast_ids.add(_img.id)
# # #                             _tag_fast_results.append({
# # #                                 "id": _img.id,
# # #                                 "filename": _img_url(_img.filename) or "",
# # #                                 "score": 97.0,
# # #                                 "timestamp": _img.timestamp.isoformat() if _img.timestamp else None,
# # #                                 "caption_short": _img.caption_short or "",
# # #                                 "person_count": _img.person_count or 0,
# # #                                 "dominant_emotion": _img.dominant_emotion or "",
# # #                                 "quality_level": _img.quality_level or "",
# # #                                 "quality_score": _img.quality_score or 0,
# # #                                 "aesthetic_score": _img.aesthetic_score or 0,
# # #                                 "user_tags": _user_tags(_img),
# # #                                 "photo_note": getattr(_img, "photo_note", "") or "",
# # #                                 "is_favorite": bool(_img.is_favorite),
# # #                                 "scene_label": _img.scene_label or "",
# # #                                 "width": _img.width, "height": _img.height,
# # #                             })
# # #                 logger.info(f"🏷️ Tag fast-path: matched tags {_matched_tags} → {len(_tag_fast_results)} images")
# # #         finally:
# # #             _db_tag.close()
# # #     except Exception as _tag_err:
# # #         logger.warning(f"Tag fast-path failed (non-fatal): {_tag_err}")

# # #     # ── Object-detection label fast-path ──────────────────────────────────
# # #     # Check if any query word directly maps to a COCO-detected object.
# # #     # scene_label contains comma-separated Faster R-CNN detections (e.g. "cat, person").
# # #     # This is far more reliable than CLIP for specific animals/objects.
# # #     QUERY_TO_COCO = {
# # #         "cat":"cat","cats":"cat","kitten":"cat","kitty":"cat","kittens":"cat",
# # #         "dog":"dog","dogs":"dog","puppy":"dog","puppies":"dog","pup":"dog",
# # #         "horse":"horse","horses":"horse","pony":"horse",
# # #         "bird":"bird","birds":"bird","parrot":"bird","eagle":"bird","owl":"bird",
# # #         "cow":"cow","cows":"cow","bull":"cow","cattle":"cow",
# # #         "elephant":"elephant","bear":"bear","zebra":"zebra","giraffe":"giraffe",
# # #         "sheep":"sheep","goat":"sheep","deer":"deer","rabbit":"rabbit",
# # #         "soccer":"sports ball","football":"sports ball","sport":"sports ball",
# # #         "cricket":"baseball bat","baseball":"baseball bat",
# # #         "tennis":"tennis racket","racket":"tennis racket",
# # #         "surfing":"surfboard","surf":"surfboard","skateboard":"skateboard",
# # #         "skiing":"skis","ski":"skis","snowboard":"snowboard",
# # #         "car":"car","cars":"car","truck":"truck","bus":"bus",
# # #         "motorcycle":"motorcycle","bicycle":"bicycle","bike":"bicycle",
# # #         "boat":"boat","ship":"boat","train":"train",
# # #         "airplane":"airplane","plane":"airplane","aircraft":"airplane",
# # #         "pizza":"pizza","cake":"cake","sandwich":"sandwich",
# # #         "apple":"apple","banana":"banana","orange":"orange",
# # #         "bottle":"bottle","cup":"cup","wine":"wine glass",
# # #         "laptop":"laptop","keyboard":"keyboard","mouse":"mouse",
# # #         "phone":"cell phone","cellphone":"cell phone",
# # #         "chair":"chair","sofa":"couch","couch":"couch","bed":"bed",
# # #         "tv":"tv","television":"tv","monitor":"tv",
# # #         "book":"book","books":"book","clock":"clock","vase":"vase",
# # #         "scissors":"scissors","knife":"knife","fork":"fork","spoon":"spoon",
# # #     }
# # #     _q_lower_words = set(query.lower().split())
# # #     _coco_targets = set()
# # #     for _w in _q_lower_words:
# # #         if _w in QUERY_TO_COCO:
# # #             _coco_targets.add(QUERY_TO_COCO[_w])
    
# # #     if _coco_targets and not should_search_people:
# # #         # Direct object detection search — bypass CLIP entirely for these queries
# # #         _db_od = SessionLocal()
# # #         try:
# # #             _od_results = []
# # #             _seen_od    = set()   # dedup by image id
# # #             _seen_fn    = set()   # dedup by filename (catches re-indexed duplicates)

# # #             # Broad animal vocabulary used by BLIP captions — used below to
# # #             # verify that the COCO model's scene_label isn't a misclassification.
# # #             _BROAD_ANIMALS = {
# # #                 "horse","cow","dog","cat","bird","elephant","tiger","lion","bear",
# # #                 "wolf","deer","sheep","goat","pig","rabbit","snake","fish","chicken",
# # #                 "duck","frog","fox","monkey","giraffe","zebra","panda","otter",
# # #                 "squirrel","raccoon","hamster","mouse","rat","kitten","puppy","calf",
# # #                 "foal","piglet","lamb","cub","chick","parrot","penguin","owl","bat",
# # #                 "seal","beaver","meerkat","flamingo","peacock","cheetah","leopard",
# # #                 "jaguar","gorilla","koala","hedgehog","iguana","chameleon","scorpion",
# # #                 "jellyfish","octopus","lobster","crab","shrimp","bison","buffalo",
# # #                 "moose","elk","reindeer","caribou","hyena","wildebeest","gazelle",
# # #             }

# # #             def _od_row(img, score):
# # #                 return {
# # #                     "id": img.id,
# # #                     "filename": _img_url(img.filename),
# # #                     "score": score,
# # #                     "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                     "caption_short": img.caption_short or "",
# # #                     "person_count": img.person_count or 0,
# # #                     "dominant_emotion": img.dominant_emotion or "",
# # #                     "quality_level": img.quality_level or "",
# # #                     "quality_score": img.quality_score or 0,
# # #                     "aesthetic_score": img.aesthetic_score or 0,
# # #                     "user_tags": _user_tags(img),
# # #                     "photo_note": getattr(img, "photo_note", "") or "",
# # #                     "is_favorite": bool(img.is_favorite),
# # #                 }

# # #             for _img in _live(_db_od).all():
# # #                 fn = _img_url(_img.filename) or ""
# # #                 # ── BUGFIX: dedup by filename to catch re-indexed duplicates ──
# # #                 if _img.id in _seen_od or fn in _seen_fn:
# # #                     continue

# # #                 # ── BUGFIX: also match via user_tags (manual tags bypass OD) ──
# # #                 _img_tags = {t.strip().lower() for t in _user_tags(_img)}
# # #                 is_tag_match = bool(_img_tags & _coco_targets)

# # #                 # ── scene_label match (original OD path) ─────────────────────
# # #                 is_od_match = False
# # #                 _match_score = 0
# # #                 if _img.scene_label:
# # #                     _detected = {x.strip().lower() for x in _img.scene_label.split(",")}
# # #                     _match_score = sum(1 for t in _coco_targets if t in _detected)
# # #                     is_od_match = _match_score > 0

# # #                 if not is_od_match and not is_tag_match:
# # #                     continue

# # #                 # ── BUGFIX: caption-verify OD matches to filter COCO misclassifications ──
# # #                 # Faster R-CNN doesn't know fox/otter/squirrel/rabbit/panda (not COCO
# # #                 # classes). It often mislabels them as "cat" or "dog". BLIP captions
# # #                 # are far more accurate — if the caption names a DIFFERENT animal,
# # #                 # the scene_label detection is a false positive. Skip it.
# # #                 # Exception: user-tagged images are always trusted.
# # #                 if is_od_match and not is_tag_match:
# # #                     _cap = (_img.caption_short or "").lower()
# # #                     if _cap:
# # #                         _cap_toks_raw = set(_cap.split())
# # #                         def _stem_tok(w):
# # #                             if len(w) > 4 and w.endswith("ses"): return w[:-2]
# # #                             if len(w) > 4 and w.endswith("ies"): return w[:-3]+"y"
# # #                             if len(w) > 3 and w.endswith("es"):  return w[:-2]
# # #                             if len(w) > 3 and w.endswith("s"):   return w[:-1]
# # #                             return w
# # #                         _cap_toks = _cap_toks_raw | {_stem_tok(w) for w in _cap_toks_raw}
# # #                         _cap_animals = _BROAD_ANIMALS & _cap_toks
# # #                         if _cap_animals and not (_cap_animals & _coco_targets):
# # #                             logger.info(
# # #                                 f"🚫 OD-filter: id={_img.id} caption has {_cap_animals}, "
# # #                                 f"not {_coco_targets} — skipping misclassification"
# # #                             )
# # #                             continue
# # #                     else:
# # #                         # Caption empty/missing — fall back to scene_label cross-check.
# # #                         # If scene_label contains ONLY human labels and no animal,
# # #                         # the "cat"/"dog" detection is almost certainly a false positive.
# # #                         _sl_raw = (_img.scene_label or "").lower()
# # #                         _sl_toks = set(_sl_raw.replace(",", " ").split())
# # #                         _sl_animals = _BROAD_ANIMALS & _sl_toks
# # #                         _HUMAN_SL   = {"person", "man", "woman", "boy", "girl", "people"}
# # #                         _sl_humans  = _HUMAN_SL & _sl_toks
# # #                         if _sl_humans and not _sl_animals:
# # #                             logger.info(
# # #                                 f"🚫 OD-filter (no caption): id={_img.id} scene_label has "
# # #                                 f"only humans {_sl_humans}, not {_coco_targets} — skipping"
# # #                             )
# # #                             continue

# # #                 # FIX Bug2: caption exists, no animals detected, but query IS for an animal
# # #                 # e.g. "two men in suits" for a "horse" query → exclude
# # #                 if is_od_match and not is_tag_match and _cap:
# # #                     _cap_toks_check = {w.rstrip("s") if len(w) > 3 else w
# # #                                        for w in set(_cap.split())} | set(_cap.split())
# # #                     _cap_animals_check = _BROAD_ANIMALS & _cap_toks_check
# # #                     _animal_targets = _coco_targets & {
# # #                         "cat","dog","horse","bird","cow","elephant","bear",
# # #                         "zebra","giraffe","sheep","deer","rabbit","horse",
# # #                     }
# # #                     if _animal_targets and not _cap_animals_check:
# # #                         # Caption has NO animal — check if it's a human-only image
# # #                         _HUMAN_CAP = {"man","woman","person","girl","boy","people","men",
# # #                                       "women","actor","actress","player","character","he",
# # #                                       "she","they","guy","lady","child","adult"}
# # #                         if _HUMAN_CAP & _cap_toks_check:
# # #                             logger.info(
# # #                                 f"🚫 OD-filter (human-only caption): id={_img.id} "
# # #                                 f"caption '{_cap[:60]}' has no animal for {_animal_targets}"
# # #                             )
# # #                             continue

# # #                 # FIX Bug4: non-animal OD targets (car, truck, etc.) with unrelated captions
# # #                 # e.g. "thor and thor in the avengers movie" for a "red car" query → exclude
# # #                 if is_od_match and not is_tag_match and _cap:
# # #                     _VEHICLE_COCO = {"car","truck","bus","motorcycle","bicycle","boat",
# # #                                      "airplane","train"}
# # #                     _vehicle_targets = _coco_targets & _VEHICLE_COCO
# # #                     if _vehicle_targets:
# # #                         _VEHICLE_WORDS = {"car","truck","bus","motorcycle","bike","bicycle",
# # #                                           "vehicle","driving","road","highway","parking",
# # #                                           "train","boat","ship","airplane","plane","suv",
# # #                                           "sedan","sports","jeep","porsche","ferrari",
# # #                                           "bmw","mercedes","driving","auto","automobile"}
# # #                         _FICTIONAL_WORDS = {"marvel","avengers","thor","ironman","iron",
# # #                                             "spiderman","batman","superman","hulk","anime",
# # #                                             "cartoon","movie","film","fictional","hero",
# # #                                             "villain","superhero","comic","bust","armor",
# # #                                             "avenger","character","suit"}
# # #                         _cap_set_v = set(_cap.split())
# # #                         _has_vehicle_word = bool(_VEHICLE_WORDS & _cap_set_v)
# # #                         _has_fictional    = bool(_FICTIONAL_WORDS & _cap_set_v)
# # #                         if not _has_vehicle_word and _has_fictional:
# # #                             logger.info(
# # #                                 f"🚫 OD-filter (fictional caption for vehicle): id={_img.id} "
# # #                                 f"caption '{_cap[:60]}'"
# # #                             )
# # #                             continue

# # #                 score = 92.0 if is_tag_match else round(85.0 + _match_score * 5, 1)
# # #                 _seen_od.add(_img.id)
# # #                 _seen_fn.add(fn)
# # #                 _od_results.append(_od_row(_img, score))

# # #             # Sort by score desc
# # #             _od_results.sort(key=lambda x: x["score"], reverse=True)

# # #             # FIX Bug1: dedup by caption text — catches same-content images uploaded
# # #             # twice (different UUID filenames, so filename-dedup misses them).
# # #             _seen_cap_text = set()
# # #             _od_deduped    = []
# # #             for _r in _od_results:
# # #                 _ct = (_r.get("caption_short") or "").strip().lower()
# # #                 if _ct and _ct in _seen_cap_text:
# # #                     logger.info(f"🔁 OD dedup: skipping duplicate caption '{_ct[:50]}'")
# # #                     continue
# # #                 if _ct:
# # #                     _seen_cap_text.add(_ct)
# # #                 _od_deduped.append(_r)
# # #             _od_results = _od_deduped

# # #             if _od_results:
# # #                 logger.info(f"✅ OD-search '{query}' → {len(_od_results)} results via scene_label/tags")
# # #                 return {
# # #                     "status": "found",
# # #                     "query": query,
# # #                     "count": len(_od_results),
# # #                     "results": _od_results[:top_k],
# # #                 }
# # #         finally:
# # #             _db_od.close()
# # #         # OD found nothing → try caption text search for the animal/object words
# # #         _caption_results = []
# # #         _seen_cap = set()
# # #         # Search caption text for query words directly
# # #         _search_terms = list(_q_lower_words) + [v for k,v in QUERY_TO_COCO.items() if k in _q_lower_words]
# # #         _search_terms = list(set(_search_terms))
# # #         _db_cap = SessionLocal()
# # #         try:
# # #             for _img in _live(_db_cap).all():
# # #                 # Search caption, scene_label, AND user tags
# # #                 _cap_lower = (_img.caption_short or "").lower()
# # #                 _det_lower = (_img.caption_detailed or "").lower()
# # #                 _scene_lower = (_img.scene_label or "").lower()
# # #                 _tags_lower = " ".join(_user_tags(_img)).lower()
# # #                 _haystack = " ".join(filter(None,[_cap_lower, _det_lower, _scene_lower, _tags_lower]))
# # #                 # Check if any search term appears anywhere
# # #                 if _haystack and any(t in _haystack for t in _search_terms) and _img.id not in _seen_cap:
# # #                     _seen_cap.add(_img.id)
# # #                     _caption_results.append({
# # #                         "id": _img.id,
# # #                         "filename": _img_url(_img.filename),
# # #                         "score": 80.0,
# # #                         "timestamp": _img.timestamp.isoformat() if _img.timestamp else None,
# # #                         "caption_short": _img.caption_short or "",
# # #                         "person_count": _img.person_count or 0,
# # #                         "dominant_emotion": _img.dominant_emotion or "",
# # #                         "quality_level": _img.quality_level or "",
# # #                         "quality_score": _img.quality_score or 0,
# # #                         "aesthetic_score": _img.aesthetic_score or 0,
# # #                         "user_tags": _user_tags(_img),
# # #                         "photo_note": getattr(_img, "photo_note", "") or "",
# # #                         "is_favorite": bool(_img.is_favorite),
# # #                     })
# # #         finally:
# # #             _db_cap.close()
# # #         if _caption_results:
# # #             logger.info(f"✅ Caption-search '{query}' → {len(_caption_results)} results")
# # #             return {
# # #                 "status": "found", "query": query,
# # #                 "count": len(_caption_results),
# # #                 "results": sorted(_caption_results, key=lambda x: x["score"], reverse=True)[:top_k]
# # #             }
# # #         logger.info(f"⚠️ OD+Caption search found nothing for {_search_terms}, falling back to CLIP")

# # #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# # #     if query_emb is None or search_engine.index is None:
# # #         return {"status": "error", "message": "No images indexed. Upload some photos first."}
# # #     query_lower  = query.lower()
# # #     # q_tokens_pre already defined above (before should_search_people pre-compute)

# # #     # For person-name queries, clear sig_words so CLIP scoring
# # #     # doesn't give caption bonus to irrelevant images (e.g. pig-in-sink for "Sadie Sink")
# # #     sig_words    = [] if should_search_people else [w for w in processed_query.lower().split() if len(w) > 2]
# # #     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in query_lower]

# # #     # ── Compound word expansion ────────────────────────────────────────────────
# # #     # BLIP captions write "iron man" (two words) but users search "ironman" (one).
# # #     # Expand known compounds so all keyword paths find the image.
# # #     _COMPOUND_SPLITS = {
# # #         "ironman":     ["iron","man"],      "spiderman":    ["spider","man"],
# # #         "batman":      ["bat","man"],       "superman":     ["super","man"],
# # #         "blackwidow":  ["black","widow"],   "antman":       ["ant","man"],
# # #         "warmachine":  ["war","machine"],   "blackpanther": ["black","panther"],
# # #         "ironheart":   ["iron","heart"],    "deadpool":     ["dead","pool"],
# # #         "daredevil":   ["dare","devil"],    "spiderverse":  ["spider","verse"],
# # #     }
# # #     if sig_words:
# # #         _exp = []
# # #         for _sw in sig_words:
# # #             _exp.append(_sw)
# # #             if _sw in _COMPOUND_SPLITS:
# # #                 _exp.extend(_COMPOUND_SPLITS[_sw])
# # #         sig_words = list(dict.fromkeys(_exp))

# # #     # Also expand the processed_query itself so CLIP gets "iron man" instead of "ironman"
# # #     _pq_words = processed_query.lower().split()
# # #     _pq_expanded = []
# # #     for _pw in _pq_words:
# # #         if _pw in _COMPOUND_SPLITS:
# # #             _pq_expanded.extend(_COMPOUND_SPLITS[_pw])
# # #         else:
# # #             _pq_expanded.append(_pw)
# # #     if _pq_expanded != _pq_words:
# # #         processed_query = " ".join(_pq_expanded)
# # #         logger.info(f"🔤 Compound expand: '{query}' → CLIP query='{processed_query}'")
# # #     db = SessionLocal()
# # #     try:
# # #         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k)
# # #         # ── Person / actor name search ─────────────────────────────────────
# # #         # ── Person name search — only for name-like queries ────────────────
# # #         # CRITICAL: Only run if query looks like a person name, NOT for
# # #         # descriptive queries like "woman in black dress".
# # #         # Rules:
# # #         #   - Must have at least one word >= 4 chars that isn't a common word
# # #         #   - Must not be a purely descriptive query (too many common words)
# # #         #   - Person name matching uses WHOLE WORD matching only (not substring)

# # #         DESCRIPTOR_WORDS = {
# # #             # articles / prepositions / conjunctions
# # #             "a","an","the","in","on","at","of","for","with","by","to","from",
# # #             "into","onto","over","under","above","below","near","next","beside",
# # #             "and","or","but","not","nor","yet","so","both","either","neither",
# # #             # verbs
# # #             "is","are","was","were","be","been","being","has","have","had",
# # #             "do","does","did","will","would","could","should","may","might",
# # #             "can","wearing","holding","standing","sitting","posing","looking",
# # #             "running","walking","smiling","laughing","dancing","jumping","riding",
# # #             # common adjectives
# # #             "long","short","tall","small","big","large","little","young","old",
# # #             "beautiful","pretty","lovely","nice","great","black","white","red",
# # #             "blue","green","yellow","orange","purple","pink","brown","gray",
# # #             "dark","light","bright","dark","shiny","fluffy","cute","funny",
# # #             # people descriptors (not names)
# # #             "woman","man","girl","boy","lady","guy","person","people","female",
# # #             "male","human","child","adult","baby","teenager","player","actor",
# # #             # scene/object words — CRITICAL: prevents these being treated as names
# # #             "photo","image","picture","standing","next","another","front","back",
# # #             "window","field","camera","dress","suit","shirt","jacket","hair",
# # #             "wearing","sitting","holding","looking","posing","smiling",
# # #             # common household/body nouns that appear in captions
# # #             "sink","table","chair","floor","wall","door","room","house","home",
# # #             "hand","face","head","hair","eyes","mouth","nose","body","back",
# # #             "water","grass","tree","flower","rock","snow","sand","road","street",
# # #             "book","phone","glass","bowl","plate","ball","box","bag","hat",
# # #             "food","cake","pizza","coffee","wine","beer","milk","soup",
# # #             "park","beach","mountain","forest","city","stage","roof","pool",
# # #             "movie","film","show","song","music","sport","game","team","show",
# # #             "scene","time","year","day","night","morning","evening","world",
# # #         }

# # #         raw_q      = query.lower().strip()
# # #         q_tokens   = raw_q.split()
# # #         # Words that could plausibly be part of a name (long, not descriptive)
# # #         # For short queries (≤3 words like "tom holland"), allow 3-char words
# # #         min_name_len = 3 if len(q_tokens) <= 3 else 4
# # #         name_candidates = [
# # #             w for w in q_tokens
# # #             if len(w) >= min_name_len
# # #             and w not in DESCRIPTOR_WORDS
# # #             and w.isalpha()
# # #         ]

# # #         descriptor_count = sum(1 for w in q_tokens if w in DESCRIPTOR_WORDS)
# # #         is_descriptive   = descriptor_count > len(q_tokens) * 0.5   # strict: >50% descriptors needed
# # #         should_search_people = bool(name_candidates) and not is_descriptive

# # #         vector_ids   = {r["id"] for r in results}
# # #         people_rows  = db.query(Person).all() if should_search_people else []

# # #         # ── 1. Match against named person clusters ────────────────────────
# # #         matched_people = []
# # #         if should_search_people:
# # #             for p in people_rows:
# # #                 pname_raw = (p.name or "").strip()
# # #                 pn        = pname_raw.lower()
# # #                 if not pn:
# # #                     continue
# # #                 # Skip default "Person N" labels
# # #                 if re.match(r"^person\s+\d+$", pn):
# # #                     continue
# # #                 # Skip OCR garbage: all-caps multi-word, or contains digits,
# # #                 # or contains non-alpha chars (PETRONAS INEQS, Hak IINDIA, etc.)
# # #                 if re.search(r"\d", pname_raw):
# # #                     continue   # contains digit → OCR garbage
# # #                 p_words = pn.split()
# # #                 # If all words are UPPERCASE in original → likely OCR label not a name
# # #                 if all(w == w.upper() and len(w) > 1 for w in pname_raw.split()):
# # #                     continue
# # #                 # Whole-word matching ONLY — "in" must NOT match inside "india"
# # #                 # Use word boundary: check if name word appears as standalone token
# # #                 p_parts = [pt for pt in p_words if len(pt) >= 3]
# # #                 query_tokens_set = set(q_tokens)
# # #                 matched = False
# # #                 for pt in p_parts:
# # #                     if pt in query_tokens_set:        # exact token match
# # #                         matched = True; break
# # #                 for nc in name_candidates:
# # #                     if nc in p_words:                 # query name-word in person name
# # #                         matched = True; break
# # #                     if pn == nc or nc == pn:
# # #                         matched = True; break
# # #                     # Allow partial first/last name: "vijay" matches "vijay kumar"
# # #                     if any(nc == pw for pw in p_words):
# # #                         matched = True; break
# # #                 if matched:
# # #                     matched_people.append(p)

# # #             # ── 2. Caption / OCR / VQA text search (works WITHOUT renaming) ──
# # #             # Only run if face-cluster lookup found no results yet
# # #             # (avoids flooding with false positives from common words in OCR)
# # #             COMMON_WORDS = {
# # #                 "the","and","for","are","but","not","you","all","any","can",
# # #                 "had","her","was","one","our","out","day","get","has","him",
# # #                 "his","how","its","may","new","now","old","see","two","way",
# # #                 "who","did","man","men","two","use","she","him","this","that",
# # #                 "with","from","they","have","more","will","home","also","into",
# # #                 "over","time","very","when","your","come","here","just","like",
# # #                 "long","make","many","most","some","them","than","then","these",
# # #                 "well","were","what","each","much","both","been","only","same",
# # #                 "india","team","cup","match","player","game","year","last",
# # #             }
# # #             # Only do caption search for proper-noun-looking queries
# # #             # (single words >= 4 chars that aren't common English words)
# # #             search_words = [w for w in name_candidates if len(w) >= 4 and w not in COMMON_WORDS]
# # #             if search_words and not matched_people:
# # #                 for img in _live(db).all():
# # #                     if img.id in vector_ids:
# # #                         continue
# # #                     vqa_person = ""
# # #                     try:
# # #                         vqa_d = json.loads(img.caption_vqa or "{}")
# # #                         vqa_person = vqa_d.get("person", "") or vqa_d.get("subject", "")
# # #                     except Exception:
# # #                         pass
# # #                     hay = " ".join(filter(None, [
# # #                         img.caption_short or "",
# # #                         img.caption_detailed or "",
# # #                         img.ocr_text_enhanced or "",
# # #                         img.scene_label or "",
# # #                         vqa_person,
# # #                         " ".join(_user_tags(img)),
# # #                     ])).lower()
# # #                     # Require ALL search words to match (avoids false positives)
# # #                     if all(w in hay for w in search_words):
# # #                         results.append({
# # #                             "id": img.id, "filename": _img_url(img.filename),
# # #                             "score": 65.0,
# # #                             "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                             "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # #                             "person_count": img.person_count or 0,
# # #                             "caption_short": img.caption_short or "",
# # #                             "user_tags": _user_tags(img),
# # #                             "photo_note": getattr(img, "photo_note", "") or "",
# # #                         })
# # #                         vector_ids.add(img.id)

# # #             if matched_people:
# # #                 # CRITICAL: When we found a person match, REPLACE CLIP results entirely.
# # #                 # CLIP results for name queries are unreliable (e.g. "sadie sink" → pig-in-sink).
# # #                 # Only return face-cluster matches + caption/OCR name matches.
# # #                 person_results = []
# # #                 person_ids_seen = set()

# # #                 for person in matched_people:
# # #                     logger.info(f"🧑 Searching for person: '{person.name}' (id={person.id})")
# # #                     face_records = db.query(DBFace).filter(DBFace.person_id == person.id).all()
# # #                     face_img_ids = list({f.image_id for f in face_records if f.image_id})
# # #                     logger.info(f"  → {len(face_records)} face records, {len(face_img_ids)} unique image_ids")

# # #                     # 1. Face-cluster matched images (highest confidence = 95)
# # #                     if face_img_ids:
# # #                         for img in _live(db).filter(DBImage.id.in_(face_img_ids)).all():
# # #                             if img.id not in person_ids_seen:
# # #                                 person_results.append({
# # #                                     "id": img.id, "filename": _img_url(img.filename),
# # #                                     "score": 95.0,
# # #                                     "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                                     "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # #                                     "person_count": img.person_count or 0,
# # #                                     "caption_short": img.caption_short or "",
# # #                                     "user_tags": _user_tags(img),
# # #                                     "photo_note": getattr(img, "photo_note", "") or "",
# # #                                     "matched_person": person.name,
# # #                                 })
# # #                                 person_ids_seen.add(img.id)

# # #                     # 2. Caption/OCR text search for the FULL NAME as a phrase only
# # #                     # Use full name phrase — prevents "sink" matching pig captions
# # #                     name_kw = person.name.lower()
# # #                     name_parts = name_kw.split()  # ["sadie", "sink"]
# # #                     for img in _live(db).all():
# # #                         if img.id in person_ids_seen:
# # #                             continue
# # #                         hay = " ".join(filter(None, [
# # #                             img.caption_short or "", img.caption_detailed or "",
# # #                             img.caption_vqa or "", img.ocr_text_enhanced or "",
# # #                             img.scene_label or ""
# # #                         ])).lower()
# # #                         # Must contain FULL NAME as phrase OR all parts together
# # #                         # NOT individual words (prevents "sink" matching pig captions)
# # #                         if name_kw in hay or (len(name_parts) > 1 and all(p in hay for p in name_parts)):
# # #                             # Extra check: if name has common words (like "sink"),
# # #                             # require the rare part to appear near human context
# # #                             person_results.append({
# # #                                 "id": img.id, "filename": _img_url(img.filename),
# # #                                 "score": 75.0,
# # #                                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# # #                                 "person_count": img.person_count or 0,
# # #                                 "caption_short": img.caption_short or "",
# # #                                 "user_tags": _user_tags(img),
# # #                                 "photo_note": getattr(img, "photo_note", "") or "",
# # #                                 "matched_person": person.name,
# # #                             })
# # #                             person_ids_seen.add(img.id)

# # #                     if not face_img_ids:
# # #                         logger.warning(f"  ⚠️ No face records for '{person.name}' — only caption search. Run Re-index AI.")

# # #                 if person_results:
# # #                     results = sorted(person_results, key=lambda x: x["score"], reverse=True)
# # #                     return {"status": "found", "query": query, "count": len(results), "results": results[:top_k]}
# # #                 else:
# # #                     # Person found in DB but no photos linked yet — tell user to re-index
# # #                     names = ", ".join(p.name for p in matched_people)
# # #                     return {
# # #                         "status": "not_found",
# # #                         "message": f"Found person '{names}' but no photos linked. Click Re-index AI to rebuild face clusters.",
# # #                         "people_suggestions": [{"id": p.id, "name": p.name, "cover": None} for p in matched_people],
# # #                     }

# # #         # ── No person found in DB for a name query → try content search first ──
# # #         if should_search_people and not matched_people:
# # #             # Before giving up, attempt 3 fallbacks in priority order:
# # #             # 1. Tag fast-path results (user manually tagged this image)
# # #             # 2. Caption/OCR keyword search (works for ironman, thor, spider, etc.)
# # #             # 3. Only if both fail → show "not found" with people suggestions

# # #             # ── Fallback 1: tags ──────────────────────────────────────────────
# # #             if _tag_fast_results:
# # #                 logger.info(f"🏷️ Person-path fallback → {len(_tag_fast_results)} tag results for '{query}'")
# # #                 return {
# # #                     "status": "found", "query": query,
# # #                     "count": len(_tag_fast_results),
# # #                     "results": _tag_fast_results[:top_k],
# # #                 }

# # #             # ── Fallback 2: caption + OCR keyword search ──────────────────────
# # #             # Covers: "ironman", "thor", "spider", "kpop", "user" (OCR), etc.
# # #             # Use the original query words (NOT processed_query which strips non-CLIP terms)
# # #             _kw_query_words = [w for w in query.lower().split() if len(w) > 2]
# # #             _kw_results_fb  = []
# # #             _kw_seen_fb     = set()
# # #             _kw_seen_cap_fb = set()
# # #             if _kw_query_words:
# # #                 for _img_fb in _live(db).all():
# # #                     if _img_fb.id in _kw_seen_fb:
# # #                         continue
# # #                     # Build search haystack: caption + OCR + scene_label + tags
# # #                     _fb_hay = " ".join(filter(None, [
# # #                         (_img_fb.caption_short    or "").lower(),
# # #                         (_img_fb.caption_detailed or "").lower(),
# # #                         (_img_fb.ocr_text_enhanced or "").lower(),
# # #                         (_img_fb.scene_label      or "").lower(),
# # #                         " ".join(_user_tags(_img_fb)).lower(),
# # #                     ]))
# # #                     if not _fb_hay:
# # #                         continue
# # #                     # ALL query words must appear (strict — avoids false positives)
# # #                     if all(w in _fb_hay for w in _kw_query_words):
# # #                         _ct_fb = (_img_fb.caption_short or "").strip().lower()
# # #                         if _ct_fb and _ct_fb in _kw_seen_cap_fb:
# # #                             continue
# # #                         if _ct_fb:
# # #                             _kw_seen_cap_fb.add(_ct_fb)
# # #                         _kw_seen_fb.add(_img_fb.id)
# # #                         _kw_results_fb.append({
# # #                             "id": _img_fb.id,
# # #                             "filename": _img_url(_img_fb.filename) or "",
# # #                             "score": 80.0,
# # #                             "timestamp": _img_fb.timestamp.isoformat() if _img_fb.timestamp else None,
# # #                             "caption_short": _img_fb.caption_short or "",
# # #                             "person_count": _img_fb.person_count or 0,
# # #                             "dominant_emotion": _img_fb.dominant_emotion or "",
# # #                             "quality_level": _img_fb.quality_level or "",
# # #                             "quality_score": _img_fb.quality_score or 0,
# # #                             "aesthetic_score": _img_fb.aesthetic_score or 0,
# # #                             "user_tags": _user_tags(_img_fb),
# # #                             "photo_note": getattr(_img_fb, "photo_note", "") or "",
# # #                             "is_favorite": bool(_img_fb.is_favorite),
# # #                             "scene_label": _img_fb.scene_label or "",
# # #                             "width": _img_fb.width, "height": _img_fb.height,
# # #                         })
# # #             if _kw_results_fb:
# # #                 logger.info(f"📝 Person-path caption/OCR fallback → {len(_kw_results_fb)} results for '{query}'")
# # #                 return {
# # #                     "status": "found", "query": query,
# # #                     "count": len(_kw_results_fb),
# # #                     "results": sorted(_kw_results_fb, key=lambda x: x["score"], reverse=True)[:top_k],
# # #                 }

# # #             # ── Fallback 3: truly nothing found → people suggestions ──────────
# # #             all_people = db.query(Person).all()
# # #             suggestions = []
# # #             for p in all_people:
# # #                 if re.match(r"^person\s+\d+$", (p.name or "").lower()): continue
# # #                 face_recs = db.query(DBFace).filter(DBFace.person_id == p.id).limit(1).all()
# # #                 if face_recs and face_recs[0].image_id:
# # #                     cover_img = db.query(DBImage).filter(DBImage.id == face_recs[0].image_id).first()
# # #                     if cover_img:
# # #                         suggestions.append({"id": p.id, "name": p.name, "cover": _img_url(cover_img.filename)})
# # #             return {
# # #                 "status": "not_found",
# # #                 "message": f"No photos matched '{query}'.",
# # #                 "people_suggestions": suggestions[:12],
# # #             }

# # #         # Fallback: discard any CLIP junk when person matched
# # #         if matched_people:
# # #             person_results = [r for r in results if r.get("matched_person")]
# # #             if person_results:
# # #                 results = person_results
# # #             results.sort(key=lambda x: x["score"], reverse=True)

# # #         # ── Skip keyword fallback when we already have person results ──────────
# # #         # Person name searches should ONLY return person face images, not
# # #         # generic CLIP results or keyword matches for ambiguous words like "sink"
# # #         skip_kw_fallback = bool(matched_people)

# # #         if sig_words and not skip_kw_fallback:
# # #             vector_ids = {r["id"] for r in results}
# # #             KW_STOP = {"the","a","an","in","on","at","of","for","with","by","to",
# # #                        "from","is","are","was","were","be","and","or","but","not"}
# # #             meaningful = [w for w in sig_words if w not in KW_STOP and len(w) > 2]

# # #             # ── Rebuild contradiction context for filtering ────────────────
# # #             COLOURS_KW = {"black","white","red","blue","green","yellow","orange",
# # #                           "purple","pink","brown","gray","grey","golden","silver"}
# # #             CLOTHING_KW = {"dress","suit","shirt","jacket","sari","skirt","gown",
# # #                            "uniform","coat","blouse","saree","lehenga"}
# # #             ANIMALS_KW = {"horse","cow","dog","cat","bird","fox","tiger","lion",
# # #                           "bear","wolf","pig","rabbit","fish","duck","frog","otter",
# # #                           "elephant","monkey","deer","sheep","goat","snake","chicken",
# # #                           "kitten","puppy","calf","hamster","squirrel","raccoon"}
# # #             FEMALE_KW = {"woman","girl","lady","female"}
# # #             MALE_KW   = {"man","boy","guy","male"}
# # #             q_set_kw      = set(meaningful)
# # #             q_colours_kw  = COLOURS_KW  & q_set_kw
# # #             q_clothing_kw = CLOTHING_KW & q_set_kw
# # #             q_animals_kw  = ANIMALS_KW  & q_set_kw
# # #             q_gender_kw   = (FEMALE_KW | MALE_KW) & q_set_kw

# # #             def _kw_ok(caption_text):
# # #                 """Return True if caption doesn't contradict the query."""
# # #                 cap = set(caption_text.split())
# # #                 # Stemmed cap words
# # #                 def s(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
# # #                 cap_stem = {s(w) for w in cap} | cap

# # #                 if q_animals_kw:
# # #                     cap_animals = ANIMALS_KW & cap_stem
# # #                     if cap_animals and not (cap_animals & q_animals_kw):
# # #                         return False   # different animal
# # #                     # Also: if caption has PEOPLE words but NO animals at all → exclude
# # #                     # (e.g. "man standing in a field" shouldn't appear for "dog" query)
# # #                     PEOPLE_KW = {"man","woman","person","girl","boy","people","guy",
# # #                                  "lady","male","female","player","actor","celebrity"}
# # #                     if not cap_animals and (PEOPLE_KW & cap_stem):
# # #                         return False   # people-only caption for animal query
# # #                 if q_colours_kw:
# # #                     cap_col = COLOURS_KW & cap
# # #                     contradicting = cap_col - q_colours_kw
# # #                     matching      = cap_col & q_colours_kw
# # #                     if contradicting and not matching:
# # #                         return False   # only wrong colours
# # #                 if q_clothing_kw:
# # #                     cap_cl = CLOTHING_KW & cap_stem
# # #                     if cap_cl and not (cap_cl & q_clothing_kw):
# # #                         return False   # different clothing
# # #                 if q_gender_kw:
# # #                     q_f = bool(FEMALE_KW & q_gender_kw)
# # #                     q_m = bool(MALE_KW   & q_gender_kw)
# # #                     if q_f and (MALE_KW & cap) and not (FEMALE_KW & cap):
# # #                         return False
# # #                     if q_m and (FEMALE_KW & cap) and not (MALE_KW & cap):
# # #                         return False
# # #                 return True

# # #             kw_candidates = _live(db).filter(
# # #                 DBImage.caption_short.isnot(None) | DBImage.caption_detailed.isnot(None)
# # #             ).all()
# # #             for img in kw_candidates:
# # #                 if img.id in vector_ids:
# # #                     continue
# # #                 caption_text = " ".join(filter(None,[
# # #                     img.caption_short or "", img.caption_detailed or ""
# # #                 ])).lower()
# # #                 haystack = caption_text + " " + " ".join(filter(None,[
# # #                     img.ocr_text_enhanced or "", img.scene_label or "",
# # #                     " ".join(_user_tags(img))
# # #                 ])).lower()
# # #                 if not meaningful:
# # #                     continue
# # #                 # Apply contradiction filter FIRST
# # #                 if not _kw_ok(caption_text):
# # #                     continue
# # #                 # Compound word matching: "ironman" counts as matched if both
# # #                 # "iron" AND "man" are present (even as separate words)
# # #                 def _word_in_haystack(w, hs):
# # #                     if w in hs:
# # #                         return True
# # #                     parts = _COMPOUND_SPLITS.get(w)
# # #                     if parts and all(p in hs for p in parts):
# # #                         return True
# # #                     return False
# # #                 # For 1-2 meaningful words: ALL must match
# # #                 # For 3+: majority (≥70%) must match
# # #                 matched = sum(1 for w in meaningful if _word_in_haystack(w, haystack))
# # #                 threshold = len(meaningful) if len(meaningful) <= 2 else max(2, len(meaningful) * 0.70)
# # #                 if matched >= threshold:
# # #                     results.append({
# # #                         "id": img.id, "filename": _img_url(img.filename), "score": 32.0,
# # #                         "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                         "person_count": img.person_count or 0,
# # #                         "caption_short": img.caption_short or "",
# # #                         "caption_detailed": img.caption_detailed or "",
# # #                         "quality_level": img.quality_level or "",
# # #                         "quality_score": img.quality_score or 0,
# # #                         "dominant_emotion": img.dominant_emotion or "",
# # #                         "face_emotion_count": img.face_emotion_count or 0,
# # #                         "aesthetic_score": img.aesthetic_score or 0,
# # #                         "aesthetic_rating": img.aesthetic_rating or "",
# # #                         "is_favorite": bool(img.is_favorite),
# # #                         "user_tags": _user_tags(img),
# # #                         "photo_note": getattr(img, "photo_note", "") or "",
# # #                         "scene_label": img.scene_label or "",
# # #                         "width": img.width, "height": img.height,
# # #                     })
# # #             results.sort(key=lambda x: x["score"], reverse=True)
# # #             results = results[:top_k]

# # #         elif sig_words and skip_kw_fallback:
# # #             # Person search mode: drop any CLIP results that aren't person matches
# # #             # (e.g. "sadie sink" → remove literal sink images from CLIP)
# # #             person_img_ids = {r["id"] for r in results if r.get("matched_person")}
# # #             if person_img_ids:
# # #                 results = [r for r in results if r.get("matched_person") or r["id"] in person_img_ids]
# # #             results.sort(key=lambda x: x["score"], reverse=True)
# # #             results = results[:top_k]
# # #         if not results:
# # #             # If CLIP found nothing but the tag fast-path did, return tag results directly
# # #             if _tag_fast_results:
# # #                 logger.info(f"🏷️ CLIP found nothing — returning {len(_tag_fast_results)} tag-matched results")
# # #                 return {
# # #                     "status": "found", "query": query,
# # #                     "count": len(_tag_fast_results),
# # #                     "results": _tag_fast_results[:top_k],
# # #                 }
# # #             # Return people suggestions so user can identify who to search
# # #             people_suggestions = []
# # #             if name_candidates:
# # #                 all_people = db.query(Person).all()
# # #                 for p in all_people:
# # #                     face_recs = db.query(DBFace).filter(DBFace.person_id == p.id).limit(1).all()
# # #                     if face_recs and face_recs[0].image_id:
# # #                         cover_img = db.query(DBImage).filter(DBImage.id == face_recs[0].image_id).first()
# # #                         if cover_img:
# # #                             people_suggestions.append({
# # #                                 "id": p.id,
# # #                                 "name": p.name,
# # #                                 "cover": _img_url(cover_img.filename)
# # #                             })
# # #             return {
# # #                 "status": "not_found",
# # #                 "message": f"No images matched '{query}'.",
# # #                 "people_suggestions": people_suggestions[:12],
# # #             }
# # #         # ── Merge tag fast-path results (score=97) before dedup ─────────────
# # #         # Tag-matched images that weren't already in CLIP results get prepended
# # #         # at the top so they always appear first regardless of CLIP score.
# # #         if _tag_fast_results:
# # #             existing_ids = {r["id"] for r in results}
# # #             for _tr in _tag_fast_results:
# # #                 if _tr["id"] not in existing_ids:
# # #                     results.insert(0, _tr)
# # #                     existing_ids.add(_tr["id"])
# # #                 else:
# # #                     # Already in results — boost its score to 97
# # #                     for _r in results:
# # #                         if _r["id"] == _tr["id"]:
# # #                             _r["score"] = max(_r.get("score", 0), 97.0)
# # #                             break
# # #             results.sort(key=lambda x: x["score"], reverse=True)

# # #         # ── BUGFIX: Final dedup by filename ─────────────────────────────────
# # #         # Catches re-indexed duplicate DB rows (different id, same physical file)
# # #         # that bypass the earlier id-based dedup in _score_candidates.
# # #         _seen_fn_final  = set()
# # #         _seen_cap_final = set()   # also dedup by caption — catches same-content re-uploads
# # #         _deduped_final  = []
# # #         for _r in results:
# # #             _fn  = _r.get("filename") or ""
# # #             _ct  = (_r.get("caption_short") or "").strip().lower()
# # #             if _fn and _fn in _seen_fn_final:
# # #                 continue
# # #             # Caption dedup: skip if identical caption already in results
# # #             # (keep first = highest scored; favourites get boosted by tag merge so appear first)
# # #             if _ct and _ct in _seen_cap_final:
# # #                 continue
# # #             if _fn:
# # #                 _seen_fn_final.add(_fn)
# # #             if _ct:
# # #                 _seen_cap_final.add(_ct)
# # #             _deduped_final.append(_r)
# # #         results = _deduped_final

# # #         return {"status": "found", "query": query, "count": len(results), "results": results}
# # #     finally:
# # #         db.close()

# # # @app.post("/search/describe")
# # # def search_by_description(description: str = Form(...), top_k: int = Form(20)):
# # #     if not description or not description.strip():
# # #         return {"status": "error", "message": "Description empty"}
# # #     if search_engine.index is None:
# # #         return {"status": "error", "message": "No images indexed."}
# # #     desc = _clean_query(description.strip())
# # #     prompts = [desc, f"a photo of {desc}", f"an image showing {desc}", f"a picture of {desc}", f"{desc} photograph"]
# # #     embs = [e for p in prompts for e in [search_engine.get_text_embedding(p, use_prompt_ensemble=False)] if e is not None]
# # #     if not embs:
# # #         return {"status": "error", "message": "Could not encode description."}
# # #     query_emb = np.mean(embs, axis=0).astype("float32")
# # #     norm = np.linalg.norm(query_emb)
# # #     if norm > 1e-8:
# # #         query_emb /= norm
# # #     sig_words    = [w for w in desc.lower().split() if len(w) > 2]
# # #     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in desc.lower()]
# # #     db = SessionLocal()
# # #     try:
# # #         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k)
# # #         if not results:
# # #             return {"status": "not_found", "message": f"No images matched description '{desc}'."}
# # #         return {"status": "found", "query": desc, "count": len(results), "results": results}
# # #     finally:
# # #         db.close()

# # # @app.post("/search/hybrid")
# # # async def search_hybrid(query: str = Form(""), file: UploadFile = File(None), text_weight: float = Form(0.6), image_weight: float = Form(0.4), top_k: int = Form(20)):
# # #     if search_engine.index is None:
# # #         return {"status": "error", "message": "No images indexed."}
# # #     has_text  = query.strip() != ""
# # #     has_image = file is not None and file.filename
# # #     if not has_text and not has_image:
# # #         return {"status": "error", "message": "Provide at least a query or an image."}
# # #     if has_text and has_image:
# # #         total_w = text_weight + image_weight
# # #         if total_w <= 0:
# # #             text_weight, image_weight = 0.6, 0.4
# # #             total_w = 1.0
# # #         text_weight /= total_w; image_weight /= total_w
# # #     elif has_text:
# # #         text_weight, image_weight = 1.0, 0.0
# # #     else:
# # #         text_weight, image_weight = 0.0, 1.0
# # #     text_emb = None
# # #     if has_text:
# # #         text_emb = search_engine.get_text_embedding(_clean_query(query), use_prompt_ensemble=True)
# # #     img_emb = None
# # #     if has_image:
# # #         import tempfile
# # #         ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
# # #         with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
# # #             tmp_path = tmp.name
# # #             shutil.copyfileobj(file.file, tmp)
# # #         try:
# # #             img_emb = search_engine.get_image_embedding(tmp_path)
# # #         finally:
# # #             try: os.remove(tmp_path)
# # #             except Exception: pass
# # #     if text_emb is not None and img_emb is not None:
# # #         query_emb, extra_emb = text_emb, img_emb
# # #     elif text_emb is not None:
# # #         query_emb, extra_emb = text_emb, None
# # #     elif img_emb is not None:
# # #         query_emb, extra_emb = img_emb, None; text_weight, image_weight = 1.0, 0.0
# # #     else:
# # #         return {"status": "error", "message": "Could not encode query."}
# # #     sig_words    = [w for w in query.lower().split() if len(w) > 2]
# # #     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in query.lower()]
# # #     db = SessionLocal()
# # #     try:
# # #         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k, extra_emb=extra_emb, text_weight=text_weight, image_weight=image_weight)
# # #         if not results:
# # #             return {"status": "not_found", "message": "No images matched."}
# # #         return {"status": "found", "query": query, "count": len(results), "text_weight": text_weight, "image_weight": image_weight, "results": results}
# # #     finally:
# # #         db.close()

# # # @app.post("/search/voice")
# # # def voice_search_legacy(duration: int = Form(5)):
# # #     try:
# # #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# # #         if not transcribed or not transcribed.strip():
# # #             return {"status": "error", "message": "Could not hear anything."}
# # #         result = search(query=transcribed.strip(), top_k=20)
# # #         result["transcribed"] = transcribed.strip()
# # #         return result
# # #     except Exception as e:
# # #         return {"status": "error", "message": f"Voice search failed: {str(e)}"}

# # # @app.post("/search/image")
# # # async def search_by_image(file: UploadFile = File(...), top_k: int = Form(20)):
# # #     if search_engine.index is None:
# # #         return {"status": "error", "message": "No images indexed."}
# # #     ext = os.path.splitext(file.filename or "")[1].lower()
# # #     if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
# # #         raise HTTPException(status_code=400, detail="Unsupported format.")
# # #     import tempfile
# # #     with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
# # #         tmp_path = tmp.name
# # #         shutil.copyfileobj(file.file, tmp)
# # #     try:
# # #         query_emb = search_engine.get_image_embedding(tmp_path)
# # #     finally:
# # #         try: os.remove(tmp_path)
# # #         except Exception: pass
# # #     if query_emb is None:
# # #         return {"status": "error", "message": "Could not process image."}
# # #     total = search_engine.index.ntotal
# # #     q = query_emb.reshape(1, -1).astype("float32")
# # #     faiss.normalize_L2(q)
# # #     distances, indices = search_engine.index.search(q, min(top_k * 3, total))
# # #     db = SessionLocal()
# # #     try:
# # #         results = []
# # #         for dist, idx in zip(distances[0], indices[0]):
# # #             if idx == -1: continue
# # #             score = float(dist)
# # #             if score < 0.45: break
# # #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# # #             if not img or img.is_trashed: continue
# # #             results.append({"id": img.id, "filename": _img_url(img.filename), "score": round(score * 100, 2), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None, "person_count": img.person_count or 0})
# # #         results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
# # #         if not results:
# # #             return {"status": "not_found", "message": "No visually similar images found."}
# # #         return {"status": "found", "count": len(results), "results": results}
# # #     finally:
# # #         db.close()

# # # @app.post("/search/color")
# # # def search_by_color(color: str = Form(...), top_k: int = Form(20)):
# # #     """
# # #     Color search using HSV hue matching instead of raw RGB distance.
# # #     Much more accurate — a blue sky photo scores high for 'blue' regardless
# # #     of how many green trees are also in the frame.
# # #     """
# # #     import colorsys

# # #     # ── Hue ranges in degrees (0-360) for each color ────────────────────────
# # #     # Each entry: (hue_center, hue_half_width, min_saturation, min_value)
# # #     COLOR_HSV = {
# # #         "red":    [(  0, 18, 0.35, 0.20), (360, 18, 0.35, 0.20)],  # red wraps around 0/360
# # #         "orange": [( 25, 15, 0.45, 0.25)],
# # #         "yellow": [( 55, 18, 0.40, 0.30)],
# # #         "green":  [(120, 40, 0.30, 0.15)],
# # #         "blue":   [(220, 45, 0.30, 0.15)],
# # #         "purple": [(280, 30, 0.25, 0.15)],
# # #         "pink":   [(330, 25, 0.25, 0.30)],
# # #         "white":  None,   # special: high value, low saturation
# # #         "black":  None,   # special: low value
# # #         "gray":   None,   # special: low saturation
# # #         "grey":   None,
# # #         "brown":  [( 25, 18, 0.35, 0.15)],  # like orange but lower value
# # #     }

# # #     color_key = color.strip().lower()
# # #     if color_key not in COLOR_HSV:
# # #         color_key = next((k for k in COLOR_HSV if k in color_key), None)
# # #     if not color_key:
# # #         return {"status": "error", "message": f"Unknown color '{color}'."}

# # #     def _color_score(r, g, b, ckey):
# # #         """Score 0-100 how well an RGB pixel matches the target color."""
# # #         r_, g_, b_ = (r or 0)/255.0, (g or 0)/255.0, (b or 0)/255.0
# # #         h, s, v = colorsys.rgb_to_hsv(r_, g_, b_)
# # #         hue_deg = h * 360.0

# # #         if ckey in ("white", "grey", "gray"):
# # #             # High brightness, low saturation
# # #             if ckey == "white":
# # #                 return max(0.0, (v - 0.75) / 0.25 * 100) * max(0.0, (0.25 - s) / 0.25)
# # #             else:  # gray
# # #                 sat_score = max(0.0, (0.20 - s) / 0.20)
# # #                 val_score = max(0.0, 1.0 - abs(v - 0.50) / 0.50)
# # #                 return sat_score * val_score * 100
# # #         if ckey == "black":
# # #             return max(0.0, (0.25 - v) / 0.25 * 100)

# # #         ranges = COLOR_HSV[ckey]
# # #         if not ranges:
# # #             return 0.0

# # #         best = 0.0
# # #         for entry in ranges:
# # #             hc, hw, min_s, min_v = entry
# # #             # Hue distance (circular)
# # #             diff = abs(hue_deg - hc)
# # #             if diff > 180: diff = 360 - diff
# # #             if diff > hw * 2:
# # #                 continue
# # #             hue_score  = max(0.0, 1.0 - diff / (hw * 2))
# # #             sat_score  = min(1.0, max(0.0, (s - min_s) / (1.0 - min_s + 0.01)))
# # #             val_score  = min(1.0, max(0.0, (v - min_v) / (1.0 - min_v + 0.01)))
# # #             # brown special case: penalise high-value (bright) oranges
# # #             if ckey == "brown":
# # #                 val_score *= max(0.0, 1.0 - max(0.0, v - 0.55) / 0.45)
# # #             score = hue_score * sat_score * val_score
# # #             best = max(best, score)
# # #         return round(best * 100, 2)

# # #     db = SessionLocal()
# # #     try:
# # #         images = _live(db).filter(
# # #             DBImage.avg_r != None, DBImage.avg_g != None, DBImage.avg_b != None
# # #         ).all()
# # #         if not images:
# # #             return {"status": "not_found", "message": "No color data. Upload photos first."}

# # #         scored = []
# # #         for img in images:
# # #             s = _color_score(img.avg_r, img.avg_g, img.avg_b, color_key)
# # #             if s >= 10.0:  # low threshold — HSV matching is already precise
# # #                 scored.append((s, img))

# # #         scored.sort(key=lambda x: x[0], reverse=True)
# # #         scored = scored[:top_k]

# # #         if not scored:
# # #             return {"status": "not_found", "message": f"No images matched color '{color}'."}
# # #         return {
# # #             "status": "found", "query": color, "count": len(scored),
# # #             "results": [{
# # #                 "id": img.id, "filename": _img_url(img.filename),
# # #                 "score": round(s, 2),
# # #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                 "caption_short": img.caption_short or "",
# # #                 "user_tags": _user_tags(img),
# # #                 "dominant_emotion": img.dominant_emotion,
# # #                 "person_count": img.person_count or 0,
# # #             } for s, img in scored]
# # #         }
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # REPROCESS EMOTIONS (fixes existing photos that have "neutral" default)
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.post("/reprocess-colors")
# # # def reprocess_colors():
# # #     """Recompute dominant color for all existing photos using weighted histogram."""
# # #     import colorsys as _cs2
# # #     from PIL import Image as PILImage2
# # #     db = SessionLocal()
# # #     try:
# # #         images = _live(db).all()
# # #         done = 0
# # #         for img in images:
# # #             fpath = img.original_path or os.path.join(IMAGE_DIR, img.filename)
# # #             if not os.path.exists(fpath):
# # #                 continue
# # #             try:
# # #                 pil = PILImage2.open(fpath).convert("RGB").resize((64,64))
# # #                 arr2 = np.array(pil, dtype=np.float32)
# # #                 r_v = arr2[:,:,0].flatten(); g_v = arr2[:,:,1].flatten(); b_v = arr2[:,:,2].flatten()
# # #                 wts = np.array([
# # #                     _cs2.rgb_to_hsv(r/255,g/255,b/255)[1] * _cs2.rgb_to_hsv(r/255,g/255,b/255)[2] + 0.01
# # #                     for r,g,b in zip(r_v,g_v,b_v)
# # #                 ])
# # #                 tw = wts.sum()
# # #                 img.avg_r = float(np.dot(r_v, wts)/tw)
# # #                 img.avg_g = float(np.dot(g_v, wts)/tw)
# # #                 img.avg_b = float(np.dot(b_v, wts)/tw)
# # #                 done += 1
# # #             except Exception as e:
# # #                 logger.warning(f"Color reprocess failed {img.id}: {e}")
# # #         db.commit()
# # #         return {"status": "done", "updated": done}
# # #     except Exception as e:
# # #         db.rollback(); raise HTTPException(500, str(e))
# # #     finally:
# # #         db.close()


# # # @app.post("/reprocess-emotions")
# # # async def reprocess_emotions(background_tasks: BackgroundTasks, limit: int = 0):
# # #     """
# # #     Re-detect emotions on all uploaded images that still have the default
# # #     'neutral' label or no emotion set. Runs in background so the API returns
# # #     immediately.  limit=0 means process ALL.
# # #     """
# # #     db = SessionLocal()
# # #     try:
# # #         q = _live(db)
# # #         if limit > 0:
# # #             q = q.limit(limit)
# # #         image_ids = [
# # #             (img.id, img.original_path or os.path.join(IMAGE_DIR, img.filename))
# # #             for img in q.all()
# # #         ]
# # #     finally:
# # #         db.close()

# # #     def _run_reprocess(pairs):
# # #         done = 0
# # #         for img_id, fpath in pairs:
# # #             if not os.path.exists(fpath):
# # #                 continue
# # #             try:
# # #                 # ── GATE: use InsightFace face count, not Faster R-CNN person_count ──
# # #                 # person_count = 0 for close-up portraits (no full body visible).
# # #                 # InsightFace accurately detects human faces regardless of body visibility.
# # #                 # Cat faces, car headlights, turtles → InsightFace gives 0 → skip model.
# # #                 _db_chk = SessionLocal()
# # #                 try:
# # #                     _face_count = _db_chk.query(DBFace).filter(DBFace.image_id == img_id).count()
# # #                 except Exception:
# # #                     _face_count = -1  # unknown → let it through
# # #                 finally:
# # #                     _db_chk.close()

# # #                 if _face_count == 0:
# # #                     # No real human faces → clear any previously wrong emotion label
# # #                     _db_n = SessionLocal()
# # #                     try:
# # #                         _db_n.query(DBImage).filter(DBImage.id == img_id).update({
# # #                             "dominant_emotion": "neutral", "face_emotion_count": 0,
# # #                             "emotion_data": "[]",
# # #                         }, synchronize_session=False)
# # #                         _db_n.commit()
# # #                     finally:
# # #                         _db_n.close()
# # #                     continue

# # #                 ed = emotion_detection.detect_emotions(fpath)
# # #                 # Confidence filter — same as _enrich_image
# # #                 ed = [e for e in ed if e.get("confidence", 0) >= 0.50]
# # #                 dominant = ed[0]["emotion"] if ed else "neutral"
# # #                 db2 = SessionLocal()
# # #                 try:
# # #                     db2.query(DBImage).filter(DBImage.id == img_id).update({
# # #                         "dominant_emotion":   dominant,
# # #                         "face_emotion_count": len(ed),
# # #                         "emotion_data":       _json.dumps(ed),
# # #                     }, synchronize_session=False)
# # #                     db2.commit()
# # #                     done += 1
# # #                 finally:
# # #                     db2.close()
# # #             except Exception as e:
# # #                 logger.warning(f"Emotion reprocess failed for {img_id}: {e}")
# # #         logger.info(f"✅ Emotion reprocess complete: {done}/{len(pairs)} images updated")

# # #     background_tasks.add_task(_run_reprocess, image_ids)
# # #     return {
# # #         "status": "started",
# # #         "message": f"Reprocessing emotions for {len(image_ids)} images in background",
# # #         "count": len(image_ids)
# # #     }


# # # @app.post("/reprocess-names")
# # # def reprocess_names():
# # #     """
# # #     Re-run auto-naming on ALL existing person clusters.
# # #     Resets garbage names (ALL-CAPS brands, OCR fragments, short initials)
# # #     back to "Person N", then re-runs the improved auto-naming logic.
# # #     """
# # #     db = SessionLocal()
# # #     try:
# # #         # ── Step 1: Reset bad auto-assigned names ───────────────────────────
# # #         people_all = db.query(Person).all()
# # #         reset_count = 0
# # #         for i, p in enumerate(people_all):
# # #             pn = (p.name or "").strip()
# # #             # Never touch manually typed names (user renamed them — they know best)
# # #             # We can't tell manual from auto, so use heuristics to detect garbage:
# # #             should_reset = False

# # #             if not pn or len(pn) < 2:
# # #                 should_reset = True
# # #             elif not any(c.isalpha() for c in pn):
# # #                 should_reset = True
# # #             else:
# # #                 name_words = pn.split()
# # #                 # Check every word in the name
# # #                 bad_word_count = 0
# # #                 for nw in name_words:
# # #                     nw_clean = nw.rstrip(".,!?;:")
# # #                     # ALL-CAPS word (brand name / jersey text)
# # #                     if nw_clean.isalpha() and nw_clean == nw_clean.upper() and len(nw_clean) > 2:
# # #                         bad_word_count += 1
# # #                     # Too short (< 3 chars = initials / OCR fragment)
# # #                     elif len(nw_clean) < 3:
# # #                         bad_word_count += 1
# # #                     # Known garbage word
# # #                     elif nw_clean.lower() in _BLIP_NON_NAMES:
# # #                         bad_word_count += 1
# # #                     # No vowels (pure consonant = OCR garbage)
# # #                     elif not any(c in "aeiouAEIOU" for c in nw_clean):
# # #                         bad_word_count += 1
# # #                 # If ANY word in the name is garbage → reset the whole name
# # #                 if bad_word_count > 0:
# # #                     should_reset = True

# # #             if should_reset:
# # #                 p.name = f"Person {i + 1}"
# # #                 reset_count += 1

# # #         db.commit()
# # #         logger.info(f"🔄 Reset {reset_count} bad auto-names back to defaults")

# # #         # ── Step 2: Re-run auto-naming with improved logic ──────────────────
# # #         people_all = db.query(Person).all()
# # #         person_map = {i: p.id for i, p in enumerate(people_all)}
# # #         _auto_name_people(db, person_map)
# # #         db.commit()

# # #         named = db.query(Person).filter(~Person.name.startswith("Person ")).count()
# # #         total = db.query(Person).count()
# # #         return {
# # #             "status": "done",
# # #             "reset": reset_count,
# # #             "named": named,
# # #             "total": total,
# # #             "message": f"{named}/{total} people named · {reset_count} bad names reset"
# # #         }
# # #     except Exception as e:
# # #         db.rollback()
# # #         raise HTTPException(500, str(e))
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # FACES / PEOPLE
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.get("/faces")
# # # def get_faces(person_id: int = Query(None)):
# # #     db = SessionLocal()
# # #     try:
# # #         if person_id:
# # #             person = db.query(Person).filter(Person.id == person_id).first()
# # #             if not person:
# # #                 raise HTTPException(status_code=404)
# # #             rows = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == person_id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
# # #             cover  = _img_url(rows[0].filename) if rows else None
# # #             images = [{"id": img.id, "filename": _img_url(img.filename), "thumbnail": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None} for img in rows]
# # #             return {"id": person.id, "name": person.name, "face_count": len(images), "cover": cover, "images": images, "results": images}
# # #         else:
# # #             people  = db.query(Person).all()
# # #             results = []
# # #             for p in people:
# # #                 imgs = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == p.id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
# # #                 if not imgs: continue
# # #                 results.append({"id": p.id, "name": p.name, "count": len(imgs), "cover": _img_url(imgs[0].filename)})
# # #             return {"results": results, "count": len(results)}
# # #     finally:
# # #         db.close()

# # # @app.get("/people/search")
# # # def search_people_by_name(q: str = Query(...)):
# # #     """Search people by name — returns matching persons with their photos."""
# # #     db = SessionLocal()
# # #     try:
# # #         q_lower = q.lower().strip()
# # #         if not q_lower:
# # #             return {"results": []}
# # #         all_people = db.query(Person).all()
# # #         matched = []
# # #         for p in all_people:
# # #             pn = (p.name or "").lower().strip()
# # #             if not pn or pn in ("unknown",):
# # #                 continue
# # #             parts = pn.split()
# # #             if q_lower in pn or any(q_lower in part for part in parts) or any(part in q_lower for part in parts if len(part)>=2):
# # #                 face_records = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# # #                 img_ids = list({f.image_id for f in face_records if f.image_id})
# # #                 imgs = _live(db).filter(DBImage.id.in_(img_ids)).order_by(DBImage.timestamp.desc()).all() if img_ids else []
# # #                 cover = _img_url(imgs[0].filename) if imgs else None
# # #                 matched.append({
# # #                     "id": p.id, "name": p.name, "count": len(imgs), "cover": cover,
# # #                     "results": [{"id": img.id, "filename": _img_url(img.filename),
# # #                                  "score": 92.0, "matched_person": p.name,
# # #                                  "caption_short": img.caption_short or "",
# # #                                  "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# # #                                  "user_tags": _user_tags(img),
# # #                                  "person_count": img.person_count or 0} for img in imgs]
# # #                 })
# # #         return {"query": q, "count": len(matched), "results": matched}
# # #     finally:
# # #         db.close()


# # # @app.get("/people/{person_id}")
# # # def get_person(person_id: int):
# # #     db = SessionLocal()
# # #     try:
# # #         person = db.query(Person).filter(Person.id == person_id).first()
# # #         if not person:
# # #             raise HTTPException(status_code=404)
# # #         rows = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == person_id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
# # #         cover  = _img_url(rows[0].filename) if rows else None
# # #         images = [{"id": img.id, "filename": _img_url(img.filename), "thumbnail": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None} for img in rows]
# # #         return {"id": person.id, "name": person.name, "face_count": len(images), "cover": cover, "images": images, "results": images}
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
# # #     return {"status": "no_match"}


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # ALBUMS
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.get("/albums/{album_id}")
# # # def get_album_by_id(album_id: int):
# # #     db = SessionLocal()
# # #     try:
# # #         album = db.query(Album).filter(Album.id == album_id).first()
# # #         if not album:
# # #             raise HTTPException(status_code=404, detail=f"Album {album_id} not found")
# # #         images = _live(db).filter(DBImage.album_id == album_id).order_by(DBImage.timestamp).all()
# # #         cover  = _img_url(images[0].filename) if images else None
# # #         date_str = ""
# # #         if album.start_date:
# # #             date_str = album.start_date.strftime("%b %Y")
# # #             if album.end_date and album.end_date.month != album.start_date.month:
# # #                 date_str += f" – {album.end_date.strftime('%b %Y')}"
# # #         img_list = [{"id": img.id, "filename": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None, "caption_short": img.caption_short, "ocr_text_enhanced": img.ocr_text_enhanced, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "aesthetic_score": img.aesthetic_score} for img in images]
# # #         return {"id": album.id, "title": album.title, "type": album.type, "description": album.description, "date": date_str, "cover": cover, "start_date": album.start_date.isoformat() if album.start_date else None, "end_date": album.end_date.isoformat() if album.end_date else None, "image_count": len(images), "images": img_list, "results": img_list, "thumbnails": [_img_url(img.filename) for img in images[:4]]}
# # #     finally:
# # #         db.close()

# # # @app.get("/albums")
# # # def get_albums(album_id: int = Query(None)):
# # #     db = SessionLocal()
# # #     try:
# # #         if album_id:
# # #             return get_album_by_id(album_id)
# # #         albums  = db.query(Album).all()
# # #         results = []
# # #         for a in albums:
# # #             album_images = _live(db).filter(DBImage.album_id == a.id).all()
# # #             # Show empty manual albums — user just created them
# # #             if not album_images and a.type == "event":
# # #                 continue  # hide empty auto-event albums, show empty manual ones
# # #             date_str = ""
# # #             if a.start_date:
# # #                 date_str = a.start_date.strftime("%b %Y")
# # #                 if a.end_date and a.end_date.month != a.start_date.month:
# # #                     date_str += f" – {a.end_date.strftime('%b %Y')}"
# # #             cover = _img_url(album_images[0].filename) if album_images else None
# # #             results.append({"id": a.id, "title": a.title, "type": a.type or "manual",
# # #                             "description": a.description, "date": date_str,
# # #                             "cover": cover, "count": len(album_images),
# # #                             "thumbnails": [_img_url(img.filename) for img in album_images[:4]]})
# # #         return {"results": results, "count": len(results)}
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # ALBUM CRUD
# # # # ────────────────────────────────────────────────────────────────────────────

# # # @app.post("/albums/create")
# # # def create_album(
# # #     title: str = Form(...),
# # #     description: str = Form(""),
# # #     image_ids: str = Form(""),   # comma-separated image IDs to add immediately
# # # ):
# # #     """Create a new manual album, optionally with a set of images."""
# # #     db = SessionLocal()
# # #     try:
# # #         album = Album(
# # #             title=title.strip(),
# # #             description=description.strip() or None,
# # #             type="manual",
# # #         )
# # #         db.add(album); db.flush()

# # #         added = 0
# # #         if image_ids.strip():
# # #             ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
# # #             for img in _live(db).filter(DBImage.id.in_(ids)).all():
# # #                 img.album_id = album.id
# # #                 added += 1
# # #             # Set album dates from images
# # #             imgs = _live(db).filter(DBImage.album_id == album.id).all()
# # #             timestamps = [i.timestamp for i in imgs if i.timestamp]
# # #             if timestamps:
# # #                 album.start_date = min(timestamps)
# # #                 album.end_date   = max(timestamps)

# # #         db.commit()
# # #         logger.info(f"✅ Album created: '{title}' (id={album.id}, {added} images)")
# # #         return {"status": "created", "id": album.id, "title": album.title, "image_count": added}
# # #     except Exception as e:
# # #         db.rollback(); raise HTTPException(500, str(e))
# # #     finally:
# # #         db.close()


# # # @app.delete("/albums/{album_id}/delete")
# # # def delete_album(album_id: int):
# # #     """Delete an album (photos are kept, just unlinked from album)."""
# # #     db = SessionLocal()
# # #     try:
# # #         album = db.query(Album).filter(Album.id == album_id).first()
# # #         if not album:
# # #             raise HTTPException(404, "Album not found")
# # #         db.query(DBImage).filter(DBImage.album_id == album_id).update({"album_id": None})
# # #         db.delete(album); db.commit()
# # #         return {"status": "deleted", "id": album_id}
# # #     except HTTPException: raise
# # #     except Exception as e:
# # #         db.rollback(); raise HTTPException(500, str(e))
# # #     finally:
# # #         db.close()


# # # @app.delete("/albums/empty/cleanup")
# # # def cleanup_empty_albums():
# # #     """Delete all empty manual albums (useful after accidental duplicates)."""
# # #     db = SessionLocal()
# # #     try:
# # #         manual_albums = db.query(Album).filter(Album.type == "manual").all()
# # #         deleted = []
# # #         for a in manual_albums:
# # #             count = db.query(DBImage).filter(DBImage.album_id == a.id).count()
# # #             if count == 0:
# # #                 deleted.append(a.title)
# # #                 db.delete(a)
# # #         db.commit()
# # #         return {"status": "done", "deleted": len(deleted), "titles": deleted}
# # #     except Exception as e:
# # #         db.rollback(); raise HTTPException(500, str(e))
# # #     finally:
# # #         db.close()


# # # @app.post("/albums/{album_id}/rename")
# # # def rename_album(album_id: int, title: str = Form(...), description: str = Form("")):
# # #     """Rename an album and optionally update its description."""
# # #     db = SessionLocal()
# # #     try:
# # #         album = db.query(Album).filter(Album.id == album_id).first()
# # #         if not album:
# # #             raise HTTPException(404, "Album not found")
# # #         album.title = title.strip()
# # #         if description.strip():
# # #             album.description = description.strip()
# # #         db.commit()
# # #         return {"status": "renamed", "id": album_id, "title": album.title}
# # #     except HTTPException: raise
# # #     except Exception as e:
# # #         db.rollback(); raise HTTPException(500, str(e))
# # #     finally:
# # #         db.close()


# # # @app.post("/albums/{album_id}/add-images")
# # # def add_images_to_album(album_id: int, image_ids: str = Form(...)):
# # #     """Add a set of images (comma-separated IDs) to an existing album."""
# # #     db = SessionLocal()
# # #     try:
# # #         album = db.query(Album).filter(Album.id == album_id).first()
# # #         if not album:
# # #             raise HTTPException(404, "Album not found")
# # #         ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
# # #         updated = _live(db).filter(DBImage.id.in_(ids)).update(
# # #             {"album_id": album_id}, synchronize_session=False
# # #         )
# # #         # Refresh album dates
# # #         imgs = _live(db).filter(DBImage.album_id == album_id).all()
# # #         timestamps = [i.timestamp for i in imgs if i.timestamp]
# # #         if timestamps:
# # #             album.start_date = min(timestamps)
# # #             album.end_date   = max(timestamps)
# # #         db.commit()
# # #         return {"status": "added", "album_id": album_id, "added": updated}
# # #     except HTTPException: raise
# # #     except Exception as e:
# # #         db.rollback(); raise HTTPException(500, str(e))
# # #     finally:
# # #         db.close()


# # # @app.post("/albums/{album_id}/remove-images")
# # # def remove_images_from_album(album_id: int, image_ids: str = Form(...)):
# # #     """Remove images from an album (images are kept, just unlinked)."""
# # #     db = SessionLocal()
# # #     try:
# # #         ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
# # #         _live(db).filter(DBImage.id.in_(ids), DBImage.album_id == album_id).update(
# # #             {"album_id": None}, synchronize_session=False
# # #         )
# # #         db.commit()
# # #         return {"status": "removed", "album_id": album_id}
# # #     except Exception as e:
# # #         db.rollback(); raise HTTPException(500, str(e))
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # DUPLICATES / EXPLORE / STATS
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.get("/duplicates")
# # # def get_duplicates():
# # #     db = SessionLocal()
# # #     try:
# # #         all_images = _live(db).all()
# # #         if not all_images:
# # #             return {"status": "found", "duplicate_groups": [], "total_groups": 0}
# # #         groups    = duplicate_engine.find_duplicates_fast(all_images, hamming_threshold=5)
# # #         formatted = [{"count": g["count"], "total_size": g["total_size"], "images": [{"id": img["id"], "filename": _img_url(img["filename"]), "thumbnail": _img_url(img["filename"]), "size": img["size"]} for img in g["images"]]} for g in groups]
# # #         return {"status": "found", "duplicate_groups": formatted, "total_groups": len(formatted)}
# # #     except Exception as e:
# # #         logger.error(f"Duplicates error: {e}", exc_info=True)
# # #         return {"status": "error", "message": str(e), "duplicate_groups": [], "total_groups": 0}
# # #     finally:
# # #         db.close()

# # # @app.get("/explore/random")
# # # def explore_random(count: int = Query(12)):
# # #     import random as _random
# # #     db = SessionLocal()
# # #     try:
# # #         all_ids = [row[0] for row in _live(db).with_entities(DBImage.id).all()]
# # #         if not all_ids:
# # #             return {"status": "not_found", "results": []}
# # #         sampled = _random.sample(all_ids, min(count, len(all_ids)))
# # #         imgs    = db.query(DBImage).filter(DBImage.id.in_(sampled)).all()
# # #         return {"status": "found", "count": len(imgs), "results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None, "person_count": img.person_count or 0} for img in imgs]}
# # #     finally:
# # #         db.close()

# # # @app.get("/stats")
# # # def get_stats():
# # #     import collections
# # #     db = SessionLocal()
# # #     try:
# # #         total_images    = db.query(DBImage).count()
# # #         total_faces     = db.query(DBFace).count()
# # #         total_people    = db.query(Person).count()
# # #         total_albums    = db.query(Album).count()
# # #         total_favorites = _live(db).filter(DBImage.is_favorite == True).count()
# # #         total_trashed   = db.query(DBImage).filter(DBImage.is_trashed == True).count()
# # #         indexed         = search_engine.index.ntotal if search_engine.index else 0
# # #         # AI object-detection labels (deduplicated per image)
# # #         ai_counter = collections.Counter()
# # #         for row in db.query(DBImage.scene_label).filter(DBImage.scene_label != None).all():
# # #             seen_in_image = set()
# # #             for tag in (row[0] or "").split(","):
# # #                 t = tag.strip().lower()
# # #                 if t and t not in seen_in_image:
# # #                     ai_counter[t] += 1
# # #                     seen_in_image.add(t)
# # #         top_tags = [{"tag": t, "count": c} for t, c in ai_counter.most_common(10)]

# # #         # User-defined tags
# # #         user_tag_counter = collections.Counter()
# # #         for row in db.execute(text("SELECT user_tags FROM images WHERE user_tags IS NOT NULL AND user_tags != '[]' AND (is_trashed IS NULL OR is_trashed=0)")):
# # #             try:
# # #                 import json as _j
# # #                 for t in _j.loads(row[0]): user_tag_counter[t] += 1
# # #             except Exception: pass
# # #         top_user_tags = [{"tag": t, "count": c} for t, c in user_tag_counter.most_common(20)]
# # #         COLOR_BUCKETS = {"red": (220,50,50), "orange": (230,120,40), "yellow": (220,210,50), "green": (50,160,50), "blue": (50,80,220), "purple": (120,50,180), "pink": (230,130,160), "white": (240,240,240), "black": (20,20,20), "gray": (128,128,128), "brown": (140,90,50)}
# # #         color_dist = {k: 0 for k in COLOR_BUCKETS}
# # #         for img in _live(db).filter(DBImage.avg_r != None).all():
# # #             img_rgb = np.array([img.avg_r or 0, img.avg_g or 0, img.avg_b or 0], dtype=np.float32)
# # #             best_bucket = min(COLOR_BUCKETS, key=lambda k: np.linalg.norm(img_rgb - np.array(COLOR_BUCKETS[k], dtype=np.float32)))
# # #             color_dist[best_bucket] += 1
# # #         return {"total_images": total_images, "total_faces": total_faces, "total_people": total_people, "total_albums": total_albums, "total_favorites": total_favorites, "total_trashed": total_trashed, "indexed_vectors": indexed, "top_tags": top_tags, "top_user_tags": top_user_tags, "color_distribution": sorted([{"color": k, "count": v} for k, v in color_dist.items() if v > 0], key=lambda x: x["count"], reverse=True)}
# # #     finally:
# # #         db.close()


# # # # ── Common non-name words returned by BLIP when it doesn't know the name ───
# # # _BLIP_NON_NAMES = {
# # #     # articles / pronouns / common words
# # #     "a", "an", "the", "it", "he", "she", "they", "him", "her", "them",
# # #     "his", "hers", "their", "we", "our", "you", "your", "my", "i",
# # #     # question words (critical — BLIP echoes these back)
# # #     "what", "who", "which", "where", "when", "how", "why", "is", "are",
# # #     "was", "were", "be", "been", "being", "do", "does", "did", "has",
# # #     "have", "had", "will", "would", "could", "should", "may", "might",
# # #     "can", "name", "named", "called", "known",
# # #     # generic person descriptions
# # #     "man", "woman", "person", "people", "boy", "girl", "child", "adult",
# # #     "human", "face", "faces", "subject", "individual", "individuals",
# # #     "unknown", "someone", "somebody", "anyone", "nobody", "one",
# # #     # job titles / roles (BLIP often returns these)
# # #     "celebrity", "actor", "actress", "model", "player", "star", "singer",
# # #     "politician", "athlete", "musician", "artist", "president", "director",
# # #     "manager", "leader", "officer", "official", "member", "representative",
# # #     # visual/image words (BLIP captions)
# # #     "image", "photo", "picture", "figure", "portrait", "photograph",
# # #     "this", "that", "these", "those", "here", "there",
# # #     # caption filler words BLIP uses
# # #     "main", "shows", "show", "showing", "appears", "appear", "looking",
# # #     "wearing", "holding", "standing", "sitting", "smiling",
# # #     "in", "on", "at", "of", "for", "with", "from", "by", "to",
# # #     "and", "or", "but", "not", "also", "as", "so", "then", "than",
# # #     "black", "white", "two", "three", "four", "five", "several", "many",
# # #     # Sports context — appear on jerseys, banners, scoreboards
# # #     "india", "team", "cup", "match", "captain", "wicket", "wizard",
# # #     "champion", "cricket", "football", "sport", "league", "series",
# # #     "trophy", "test", "ipl", "bcci", "odi", "t20", "final", "world",
# # #     "semi", "quarter", "squad", "player", "coach", "umpire", "referee",
# # #     # Brand/sponsor words (F1, cricket, general)
# # #     "petronas", "stake", "marlboro", "ferrari", "mercedes", "mclaren",
# # #     "alpine", "williams", "haas", "redbull", "birla", "estates", "bank",
# # #     "mutual", "fund", "limited", "pvt", "company", "corp", "inc", "ltd",
# # #     "group", "holding", "sponsor", "official", "partner",
# # #     # Common OCR fragments from Indian sports context
# # #     "ndia", "iindia", "ldcup", "ineqs", "eironas", "hak", "zon",
# # #     # Titles / honorifics misread as names
# # #     "master", "mister", "miss", "mrs", "sir", "dr", "prof", "mr",
# # #     # Generic English words that look capitalised in headlines
# # #     "new", "old", "big", "great", "good", "best", "top", "live",
# # #     "real", "true", "blue", "gold", "king", "queen", "super", "ultra",
# # # }

# # # # ── Patterns that definitively mark a word as NOT a person name ─────────────
# # # def _is_valid_name_word(w: str, allow_short: bool = False) -> bool:
# # #     """
# # #     Return True only if `w` looks like part of a real person's name.
# # #     Rejects: ALL-CAPS brands, too-short initials, non-alpha, known non-names.
# # #     """
# # #     w_clean = w.rstrip(".,!?;:'\"")
# # #     if not w_clean:
# # #         return False
# # #     # Must be purely alphabetic (no digits, no hyphens, no underscores)
# # #     if not w_clean.isalpha():
# # #         return False
# # #     # ALL-CAPS words are brand names / acronyms / jersey text, not person names
# # #     if w_clean == w_clean.upper() and len(w_clean) > 2:
# # #         return False
# # #     # Minimum length — real given names are rarely < 3 chars in isolation
# # #     min_len = 3 if allow_short else 4
# # #     if len(w_clean) < min_len:
# # #         return False
# # #     # Must start with uppercase (proper noun)
# # #     if not w_clean[0].isupper():
# # #         return False
# # #     # Must not be a known non-name
# # #     if w_clean.lower() in _BLIP_NON_NAMES:
# # #         return False
# # #     # Must contain at least one vowel (pure consonant strings = OCR garbage)
# # #     if not any(c in "aeiouAEIOU" for c in w_clean):
# # #         return False
# # #     return True


# # # def _extract_name_from_vqa(vqa_text: str) -> str:
# # #     """
# # #     Parse a BLIP VQA answer to extract a plausible person name.
# # #     Very strict — only returns something if it looks like a real proper name.
# # #     Returns empty string if uncertain.
# # #     """
# # #     if not vqa_text:
# # #         return ""
# # #     text = vqa_text.strip()

# # #     # Strip common prefixes BLIP adds
# # #     for prefix in [
# # #         "the person is ", "this is ", "it is ", "that is ",
# # #         "his name is ", "her name is ", "the name is ",
# # #         "the man is ", "the woman is ", "i think it is ",
# # #         "i think it's ", "i believe it is ", "appears to be ",
# # #         "looks like ", "this appears to be ", "the answer is ",
# # #     ]:
# # #         if text.lower().startswith(prefix):
# # #             text = text[len(prefix):]

# # #     # Take just the first 1-2 words
# # #     words = text.split()[:2]
# # #     if not words:
# # #         return ""

# # #     # Validate each word — allow short (3-char) first names only in 2-word combos
# # #     if len(words) == 2:
# # #         w1_ok = _is_valid_name_word(words[0], allow_short=True)
# # #         w2_ok = _is_valid_name_word(words[1], allow_short=False)
# # #         if w1_ok and w2_ok:
# # #             name = f"{words[0].rstrip('.,!?;:')} {words[1].rstrip('.,!?;:')}"
# # #             return name if 4 <= len(name) <= 35 else ""
# # #         if w1_ok:
# # #             w = words[0].rstrip(".,!?;:")
# # #             return w if 4 <= len(w) <= 35 else ""
# # #         return ""
# # #     else:
# # #         w = words[0].rstrip(".,!?;:")
# # #         if _is_valid_name_word(w, allow_short=False):
# # #             return w if 4 <= len(w) <= 35 else ""
# # #         return ""


# # # def _auto_name_people(db, person_map: dict):
# # #     """
# # #     For each newly created person cluster, try to auto-name them.
# # #     Strategy (strict):
# # #       1. VQA "person" answers get highest weight (3 pts each) — most reliable
# # #       2. OCR proper nouns that appear in MULTIPLE images of the same cluster get 1 pt each
# # #          (single-image OCR hits are often jersey text / background banners → ignored)
# # #       3. Only assign a name if the top candidate has >= 50% of photos voting for it
# # #          AND if it's a VQA-sourced name, OR >= 2 images (to prevent single-image OCR garbage)
# # #     """
# # #     import json as _jj
# # #     from collections import Counter

# # #     for label, person_id in person_map.items():
# # #         person = db.query(Person).filter(Person.id == person_id).first()
# # #         if not person:
# # #             continue
# # #         # Only auto-name if still using the default label
# # #         if not person.name.startswith("Person "):
# # #             continue

# # #         face_records = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# # #         img_ids = list({f.image_id for f in face_records if f.image_id})
# # #         if not img_ids:
# # #             continue

# # #         images = db.query(DBImage).filter(DBImage.id.in_(img_ids)).all()
# # #         n_images = len(images)

# # #         # Separate counters: VQA (trusted) vs OCR (needs cross-image confirmation)
# # #         vqa_counter = Counter()
# # #         ocr_counter = Counter()   # counts how many IMAGES contain each OCR name

# # #         for img in images:
# # #             # ── VQA: most reliable, use directly ────────────────────────────
# # #             if img.caption_vqa:
# # #                 try:
# # #                     vqa = _jj.loads(img.caption_vqa)
# # #                     vqa_name = _extract_name_from_vqa(vqa.get("person", ""))
# # #                     if vqa_name:
# # #                         vqa_counter[vqa_name] += 1
# # #                 except Exception:
# # #                     pass

# # #             # ── OCR: scan for proper-noun pairs and single words ─────────────
# # #             # Only count a name ONCE per image (to prevent a single jersey from
# # #             # winning just because the text appears 10 times in one photo).
# # #             ocr = img.ocr_text_enhanced or ""
# # #             if ocr:
# # #                 seen_in_this_image = set()
# # #                 words = ocr.split()
# # #                 for i, w in enumerate(words):
# # #                     # Try two-word name first
# # #                     if i + 1 < len(words):
# # #                         w2 = words[i + 1].rstrip(".,!?;:'\"")
# # #                         w1 = w.rstrip(".,!?;:'\"")
# # #                         if (_is_valid_name_word(w1, allow_short=True) and
# # #                                 _is_valid_name_word(w2, allow_short=False)):
# # #                             two_word = f"{w1} {w2}"
# # #                             if two_word not in seen_in_this_image:
# # #                                 seen_in_this_image.add(two_word)
# # #                                 ocr_counter[two_word] += 1

# # #                     # Single word fallback
# # #                     w1 = w.rstrip(".,!?;:'\"")
# # #                     if (_is_valid_name_word(w1, allow_short=False) and
# # #                             w1 not in seen_in_this_image):
# # #                         seen_in_this_image.add(w1)
# # #                         ocr_counter[w1] += 1

# # #         # ── Decision logic ────────────────────────────────────────────────────
# # #         best_name = None
# # #         best_score = 0

# # #         # VQA wins if it appears in >= 25% of images (e.g. 1 out of 4)
# # #         if vqa_counter:
# # #             top_vqa, top_vqa_count = vqa_counter.most_common(1)[0]
# # #             min_vqa = max(1, n_images * 0.25)
# # #             if top_vqa_count >= min_vqa:
# # #                 best_name  = top_vqa
# # #                 best_score = top_vqa_count * 3  # weighted

# # #         # OCR only wins if:
# # #         # (a) NO VQA name found, AND
# # #         # (b) name appears in >= 2 different images (cross-image confirmation), AND
# # #         # (c) appears in >= 40% of the cluster's images
# # #         if not best_name and ocr_counter:
# # #             top_ocr, top_ocr_count = ocr_counter.most_common(1)[0]
# # #             min_ocr_images = max(2, n_images * 0.40)
# # #             if top_ocr_count >= min_ocr_images:
# # #                 best_name  = top_ocr
# # #                 best_score = top_ocr_count

# # #         if best_name:
# # #             person.name = best_name
# # #             logger.info(f"🏷️  Auto-named Person {person_id} → '{best_name}' "
# # #                         f"(score={best_score}, vqa={dict(vqa_counter.most_common(3))}, "
# # #                         f"n_images={n_images})")
# # #         else:
# # #             top_vqa_str = str(vqa_counter.most_common(1)) if vqa_counter else "none"
# # #             top_ocr_str = str(ocr_counter.most_common(3)) if ocr_counter else "none"
# # #             logger.info(f"⚠️  No confident name for Person {person_id} "
# # #                         f"(n_images={n_images}, vqa={top_vqa_str}, top_ocr={top_ocr_str})")


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # RECLUSTER
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.post("/recluster")

# # # def recluster():
# # #     db = SessionLocal()
# # #     try:
# # #         logger.info("🔄 Recluster: saving manually assigned names before reset…")

# # #         # ── Save manually named people BEFORE wiping everything ─────────────
# # #         # For each person with a non-default name, remember which face embeddings
# # #         # (as indices into the face table) belonged to them.
# # #         saved_names = {}  # face_id_set -> name
# # #         for person in db.query(Person).all():
# # #             pname = (person.name or "").strip()
# # #             # Only keep explicitly renamed people (skip "Person N" defaults)
# # #             if re.match(r"^person\s+\d+$", pname.lower()):
# # #                 continue
# # #             if not pname or pname.lower() in ("unknown", ""):
# # #                 continue
# # #             face_ids = frozenset(
# # #                 f.id for f in db.query(DBFace).filter(DBFace.person_id == person.id).all()
# # #             )
# # #             if face_ids:
# # #                 saved_names[face_ids] = pname
# # #                 logger.info(f"💾 Saved name '{pname}' ({len(face_ids)} faces)")

# # #         logger.info(f"🔄 Recluster: clearing old assignments…")
# # #         db.query(DBFace).update({"person_id": None})
# # #         db.query(Person).delete()
# # #         db.query(DBImage).update({"album_id": None})
# # #         db.query(Album).filter(Album.type == "event").delete()
# # #         db.commit()

# # #         # ── Load face records with image dimensions for size-filtering ────────
# # #         # Background faces (tiny, far-away people) corrupt clusters — the same
# # #         # main person ends up split into multiple clusters because their face
# # #         # sometimes appears blurry/small in group shots.
# # #         # Strategy: for each image, only cluster faces that are "prominent":
# # #         #   - face area >= MIN_FACE_RATIO of image area (ignores tiny bystanders)
# # #         #   - if multiple faces pass the threshold, keep top MAX_FACES_PER_IMAGE
# # #         #     by size (supports group photos with 2-3 main subjects)
# # #         MIN_FACE_RATIO   = 0.005   # face must cover >= 0.5% of image area
# # #                                     # 1920×1080 image → face must be >= ~100×100 px
# # #         MAX_FACES_PER_IMAGE = 4    # keep at most 4 largest faces per image

# # #         # Build image dimension lookup: image_id → (width, height)
# # #         img_dims = {}
# # #         for img in db.query(DBImage.id, DBImage.width, DBImage.height).all():
# # #             if img.width and img.height:
# # #                 img_dims[img.id] = (img.width, img.height)

# # #         face_records            = db.query(DBFace).filter(DBFace.face_embedding != None).all()
# # #         embeddings, valid_faces = [], []
# # #         skipped = 0
# # #         bg_filtered = 0

# # #         # Group faces by image to allow per-image top-N selection
# # #         from collections import defaultdict
# # #         import json as _jfilt
# # #         faces_by_image = defaultdict(list)
# # #         for fr in face_records:
# # #             if not fr.image_id:
# # #                 continue
# # #             try:
# # #                 emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
# # #                 if emb.shape[0] != 512:
# # #                     skipped += 1
# # #                     continue
# # #                 # Parse bbox to get face area
# # #                 face_area_ratio = 1.0  # default: include if no bbox/dims available
# # #                 if fr.bbox and fr.image_id in img_dims:
# # #                     try:
# # #                         bbox = _jfilt.loads(fr.bbox)  # [x1, y1, x2, y2]
# # #                         x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
# # #                         face_area = max(0, x2 - x1) * max(0, y2 - y1)
# # #                         img_w, img_h = img_dims[fr.image_id]
# # #                         img_area = img_w * img_h
# # #                         face_area_ratio = face_area / img_area if img_area > 0 else 1.0
# # #                     except Exception:
# # #                         face_area_ratio = 1.0  # parse failed → include by default
# # #                 faces_by_image[fr.image_id].append((face_area_ratio, fr, emb))
# # #             except Exception as e:
# # #                 logger.warning(f"Bad embedding face {fr.id}: {e}")
# # #                 skipped += 1

# # #         # For each image: filter by MIN_FACE_RATIO, then keep top MAX_FACES_PER_IMAGE
# # #         for image_id, face_list in faces_by_image.items():
# # #             # Sort by area ratio descending (largest face first)
# # #             face_list.sort(key=lambda x: x[0], reverse=True)

# # #             # Apply area filter — but always keep at least 1 face per image
# # #             # (portrait shots where the face fills the frame pass trivially)
# # #             passed = [item for item in face_list if item[0] >= MIN_FACE_RATIO]
# # #             if not passed:
# # #                 # All faces were small — keep the single largest one anyway
# # #                 # (could be a small portrait photo shot from far away)
# # #                 passed = face_list[:1]
# # #                 logger.debug(f"🔍 Image {image_id}: all faces small, keeping largest "
# # #                              f"(ratio={face_list[0][0]:.4f})")

# # #             # Cap at MAX_FACES_PER_IMAGE
# # #             kept = passed[:MAX_FACES_PER_IMAGE]
# # #             dropped = len(face_list) - len(kept)
# # #             if dropped > 0:
# # #                 bg_filtered += dropped
# # #                 logger.debug(f"🔍 Image {image_id}: kept {len(kept)}/{len(face_list)} faces "
# # #                              f"(dropped {dropped} background faces, "
# # #                              f"min_ratio={kept[-1][0]:.4f})")

# # #             for _, fr, emb in kept:
# # #                 embeddings.append(emb)
# # #                 valid_faces.append(fr)

# # #         logger.info(f"👥 {len(embeddings)} prominent face embeddings "
# # #                     f"({skipped} invalid, {bg_filtered} background faces filtered out)")
# # #         people_count = 0
# # #         if embeddings:
# # #             labels     = face_engine.cluster_faces(embeddings)
# # #             person_map = {}
# # #             for i, label in enumerate(labels):
# # #                 if label == -1: continue
# # #                 if label not in person_map:
# # #                     p = Person(name=f"Person {label + 1}")
# # #                     db.add(p); db.flush()
# # #                     person_map[label] = p.id; people_count += 1
# # #                 valid_faces[i].person_id = person_map[label]
# # #             db.commit()
# # #             logger.info(f"✅ {people_count} people created")

# # #             # ── Restore manually assigned names to matching new clusters ────
# # #             if saved_names:
# # #                 restored = 0
# # #                 # Build map: face_id -> new person_id
# # #                 face_to_person = {}
# # #                 for face in db.query(DBFace).filter(DBFace.person_id != None).all():
# # #                     face_to_person[face.id] = face.person_id

# # #                 for old_face_ids, saved_name in saved_names.items():
# # #                     # Find which new person cluster has the most overlap with old faces
# # #                     overlap_count = {}
# # #                     for fid in old_face_ids:
# # #                         pid = face_to_person.get(fid)
# # #                         if pid:
# # #                             overlap_count[pid] = overlap_count.get(pid, 0) + 1

# # #                     if not overlap_count:
# # #                         continue
# # #                     # Best matching cluster
# # #                     best_pid = max(overlap_count, key=overlap_count.get)
# # #                     best_overlap = overlap_count[best_pid]
# # #                     # Only restore if at least 30% of faces match
# # #                     min_match = max(1, len(old_face_ids) * 0.30)
# # #                     if best_overlap >= min_match:
# # #                         person = db.query(Person).filter(Person.id == best_pid).first()
# # #                         if person and re.match(r"^person\s+\d+$", (person.name or "").lower()):
# # #                             person.name = saved_name
# # #                             restored += 1
# # #                             logger.info(f"✅ Restored name '{saved_name}' → Person {best_pid} ({best_overlap}/{len(old_face_ids)} faces matched)")
# # #                 db.commit()
# # #                 logger.info(f"🏷️  Restored {restored}/{len(saved_names)} manually assigned names")

# # #             # Auto-name remaining unnamed clusters from BLIP captions / VQA
# # #             _auto_name_people(db, person_map)
# # #             db.commit()
# # #         all_images   = db.query(DBImage).all()
# # #         albums_count = 0
# # #         if all_images:
# # #             metadata = [{"id": img.id, "lat": img.lat or 0.0, "lon": img.lon or 0.0, "timestamp": img.timestamp} for img in all_images if img.timestamp]
# # #             if metadata:
# # #                 album_labels = clustering_engine.detect_events(metadata)
# # #                 album_map    = {}
# # #                 for i, label in enumerate(album_labels):
# # #                     if label == -1: continue
# # #                     if label not in album_map:
# # #                         cluster_meta = [metadata[j] for j, l in enumerate(album_labels) if l == label]
# # #                         ts_list      = [m["timestamp"] for m in cluster_meta if m["timestamp"]]
# # #                         start_d      = min(ts_list) if ts_list else None
# # #                         end_d        = max(ts_list) if ts_list else None
# # #                         if start_d:
# # #                             if end_d and end_d.date() != start_d.date():
# # #                                 # Multi-day: "Mar 4 – 7, 2026"
# # #                                 if end_d.month == start_d.month:
# # #                                     title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%d, %Y')}"
# # #                                 else:
# # #                                     title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%b %d, %Y')}"
# # #                             else:
# # #                                 # Single day: "Mar 4, 2026"
# # #                                 title = start_d.strftime("%b %d, %Y")
# # #                         else:
# # #                             title = f"Event {label + 1}"
# # #                         new_album = Album(title=title, type="event", start_date=start_d, end_date=end_d)
# # #                         db.add(new_album); db.flush()
# # #                         album_map[label] = new_album.id; albums_count += 1
# # #                     db.query(DBImage).filter(DBImage.id == metadata[i]["id"]).update({"album_id": album_map[label]})
# # #                 db.commit()
# # #                 logger.info(f"✅ {albums_count} albums created")
# # #         return {"status": "done", "people": people_count, "albums": albums_count}
# # #     except Exception as e:
# # #         db.rollback()
# # #         logger.error(f"❌ Recluster failed: {e}", exc_info=True)
# # #         raise HTTPException(status_code=500, detail=str(e))
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # UPLOAD
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.post("/upload")
# # # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# # #     ext = os.path.splitext(file.filename)[1].lower()
# # #     if ext not in [".jpg", ".jpeg", ".png"]:
# # #         raise HTTPException(status_code=400, detail="Only JPG and PNG supported.")
# # #     filename  = f"{uuid.uuid4()}{ext}"
# # #     file_path = os.path.join(IMAGE_DIR, filename)
# # #     db        = SessionLocal()
# # #     try:
# # #         with open(file_path, "wb") as buffer:
# # #             shutil.copyfileobj(file.file, buffer)
# # #         from PIL import Image as PILImage
# # #         width = height = None
# # #         avg_r = avg_g = avg_b = 0.0
# # #         try:
# # #             img_pil       = PILImage.open(file_path).convert("RGB")
# # #             width, height = img_pil.size
# # #             arr = np.array(img_pil)
# # #             # Use dominant color (most common cluster) instead of plain average
# # #             # Resize to 64x64 for speed, then find the peak of the color histogram
# # #             import colorsys as _cs
# # #             small = np.array(img_pil.resize((64, 64)), dtype=np.float32)
# # #             r_vals = small[:,:,0].flatten()
# # #             g_vals = small[:,:,1].flatten()
# # #             b_vals = small[:,:,2].flatten()
# # #             # Weight pixels by their saturation (ignore grey/white/black background)
# # #             weights = []
# # #             for ri, gi, bi in zip(r_vals, g_vals, b_vals):
# # #                 _, s, v = _cs.rgb_to_hsv(ri/255, gi/255, bi/255)
# # #                 weights.append(s * v + 0.01)  # saturated bright pixels count more
# # #             weights = np.array(weights)
# # #             total_w = weights.sum()
# # #             avg_r = float(np.dot(r_vals, weights) / total_w)
# # #             avg_g = float(np.dot(g_vals, weights) / total_w)
# # #             avg_b = float(np.dot(b_vals, weights) / total_w)
# # #         except Exception:
# # #             pass
# # #         clip_emb = None
# # #         try:
# # #             clip_emb = search_engine.get_image_embedding(file_path)
# # #         except Exception:
# # #             pass
# # #         scene_label  = ""
# # #         person_count = 0
# # #         try:
# # #             objects      = detector_engine.detect_objects(file_path, threshold=0.5)
# # #             scene_label  = ", ".join(objects) if objects else ""
# # #             person_count = detector_engine.detect_persons(file_path)
# # #         except Exception:
# # #             pass
# # #         img_record = DBImage(filename=filename, original_path=file_path, timestamp=datetime_module.datetime.now(), width=width, height=height, avg_r=avg_r, avg_g=avg_g, avg_b=avg_b, scene_label=scene_label, person_count=person_count, ocr_text_enhanced="", ocr_keywords="[]", ocr_confidence=0.0, detected_language="en", caption_short="", caption_detailed="", caption_vqa="{}", quality_score=0.0, quality_level="Processing...", sharpness=0.0, exposure=0.0, contrast=0.0, composition=0.0, emotion_data="[]", dominant_emotion="neutral", face_emotion_count=0, aesthetic_score=0.0, aesthetic_rating="Processing...")
# # #         db.add(img_record); db.flush()
# # #         if clip_emb is not None:
# # #             try:
# # #                 if search_engine.index is None:
# # #                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
# # #                 new_vec = clip_emb.reshape(1, -1).astype("float32")
# # #                 faiss.normalize_L2(new_vec)
# # #                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id], dtype="int64"))
# # #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# # #             except Exception as e:
# # #                 logger.warning(f"FAISS update failed: {e}")
# # #         face_count = 0
# # #         try:
# # #             faces = face_engine.detect_faces(file_path)
# # #             if faces and width and height:
# # #                 img_area = width * height
# # #                 # Sort faces by area descending (largest = most prominent first)
# # #                 def _face_area(f):
# # #                     b = f["bbox"]  # [x1, y1, x2, y2]
# # #                     return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
# # #                 faces_sorted = sorted(faces, key=_face_area, reverse=True)
# # #                 # Keep only prominent faces (>= 0.5% of image area), max 4
# # #                 prominent = [f for f in faces_sorted
# # #                              if _face_area(f) / img_area >= 0.005]
# # #                 if not prominent:
# # #                     prominent = faces_sorted[:1]  # always keep largest face
# # #                 faces = prominent[:4]
# # #             for face in faces:
# # #                 emb = face["embedding"].astype(np.float32)
# # #                 db.add(DBFace(image_id=img_record.id, bbox=_json.dumps(face["bbox"]), face_embedding=emb.tobytes()))
# # #                 face_count += 1
# # #         except Exception as e:
# # #             logger.warning(f"Face detection failed: {e}")
# # #         db.commit()
# # #         image_id = img_record.id
# # #         if background_tasks is not None:
# # #             background_tasks.add_task(_enrich_image, image_id, file_path)
# # #             should_trigger_recluster(background_tasks)
# # #         else:
# # #             _enrich_image(image_id, file_path)
# # #         logger.info(f"✅ Upload done: {filename} (id={image_id})")
# # #         return {"status": "success", "id": image_id, "filename": filename, "person_count": person_count, "face_count": face_count, "quality_score": 0.0, "quality_level": "Processing...", "caption": "", "dominant_emotion": "neutral", "emotion_count": 0, "aesthetic_score": 0.0, "note": "ML enrichment running in background"}
# # #     except Exception as e:
# # #         db.rollback()
# # #         if os.path.exists(file_path): os.remove(file_path)
# # #         logger.error(f"Upload failed: {e}", exc_info=True)
# # #         raise HTTPException(status_code=500, detail=str(e))
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # REPROCESS
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.post("/reprocess_images")
# # # async def reprocess_images():
# # #     db = SessionLocal()
# # #     processed, failed = 0, 0
# # #     try:
# # #         images = db.query(DBImage).filter((DBImage.caption_short == None) | (DBImage.caption_short == "") | (DBImage.quality_level == "Processing...")).all()
# # #         if not images:
# # #             return {"success": True, "processed": 0, "failed": 0}
# # #         for img in images:
# # #             try:
# # #                 if not img.original_path or not os.path.exists(img.original_path):
# # #                     failed += 1; continue
# # #                 _enrich_image(img.id, img.original_path); processed += 1
# # #             except Exception:
# # #                 failed += 1
# # #         return {"success": True, "processed": processed, "failed": failed}
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=str(e))
# # #     finally:
# # #         db.close()


# # # # ────────────────────────────────────────────────────────────────────────────
# # # # RECAPTION — re-run BLIP on images with missing or poor captions
# # # # ────────────────────────────────────────────────────────────────────────────
# # # @app.post("/recaption")
# # # async def recaption_images(background_tasks: BackgroundTasks, force_all: bool = False):
# # #     """
# # #     Re-run BLIP captioning on images that:
# # #     - Have no caption (caption_short is NULL or empty)
# # #     - Have a placeholder/error caption
# # #     - force_all=True: re-caption every image (useful after switching BLIP model)

# # #     Runs in the background so the API returns immediately.
# # #     Progress can be checked via GET /recaption/status
# # #     """
# # #     db = SessionLocal()
# # #     try:
# # #         if force_all:
# # #             images = _live(db).all()
# # #         else:
# # #             # Only images missing captions
# # #             images = _live(db).filter(
# # #                 (DBImage.caption_short == None) |
# # #                 (DBImage.caption_short == "") |
# # #                 (DBImage.caption_short == "Processing...")
# # #             ).all()

# # #         pairs = [
# # #             (img.id, img.original_path or os.path.join(IMAGE_DIR, img.filename))
# # #             for img in images
# # #         ]
# # #         # Filter to files that actually exist
# # #         pairs = [(iid, fp) for iid, fp in pairs if os.path.exists(fp)]
# # #     finally:
# # #         db.close()

# # #     if not pairs:
# # #         return {"status": "nothing_to_do", "count": 0,
# # #                 "message": "All images already have captions. Use force_all=true to re-caption everything."}

# # #     def _run_recaption(image_pairs):
# # #         done = 0
# # #         for img_id, fpath in image_pairs:
# # #             try:
# # #                 db2 = SessionLocal()
# # #                 try:
# # #                     caption_short    = captioning_engine.generate_caption(fpath, max_length=20)
# # #                     caption_detailed = captioning_engine.generate_caption(fpath, max_length=60)
# # #                     vqa_subject = captioning_engine.answer_visual_question(
# # #                         fpath, "What is the main subject in this image?"
# # #                     )
# # #                     # Check if this image has people for VQA person question
# # #                     img_chk = db2.query(DBImage).filter(DBImage.id == img_id).first()
# # #                     vqa_person = ""
# # #                     if img_chk and (img_chk.person_count or 0) > 0:
# # #                         vqa_person = captioning_engine.answer_visual_question(fpath, "who is this?")
# # #                         if vqa_person and vqa_person.lower().strip() in {
# # #                             "man","woman","person","a man","a woman","a person","boy","girl",
# # #                             "child","human","unknown","celebrity","actor","actress","no","yes",
# # #                             "none","i don't know",
# # #                         }:
# # #                             vqa_person = ""

# # #                     db2.query(DBImage).filter(DBImage.id == img_id).update({
# # #                         "caption_short":    caption_short,
# # #                         "caption_detailed": caption_detailed,
# # #                         "caption_vqa":      _json.dumps({
# # #                             "subject": vqa_subject,
# # #                             "person":  vqa_person,
# # #                         } if (vqa_subject or vqa_person) else {}),
# # #                         "caption_timestamp": datetime_module.datetime.now(),
# # #                     }, synchronize_session=False)
# # #                     db2.commit()
# # #                     done += 1
# # #                     logger.info(f"✅ Recaptioned [{img_id}]: {caption_short}")
# # #                 finally:
# # #                     db2.close()
# # #             except Exception as e:
# # #                 logger.warning(f"⚠️ Recaption failed [{img_id}]: {e}")
# # #         logger.info(f"✅ Recaption complete: {done}/{len(image_pairs)} images")

# # #     background_tasks.add_task(_run_recaption, pairs)
# # #     return {
# # #         "status": "started",
# # #         "count": len(pairs),
# # #         "message": f"Re-captioning {len(pairs)} images in background. "
# # #                    f"{'(all images)' if force_all else '(missing captions only)'}",
# # #     }


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
# # import json as _json
# # from enhanced_ocr_engine import ocr_engine
# # from image_captioning_engine import captioning_engine
# # from quality_emotion_aesthetic_engines import (
# #     image_quality,
# #     emotion_detection,
# #     aesthetic_scoring
# # )

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
# # from fastapi.responses import FileResponse
# # from sqlalchemy import text
# # import re
# # import json
# # from api_endpoints import router as dl_router
# # from voice_route import router as voice_router
# # from features_router import router as feat_router, ensure_extra_columns   # ← only import here

# # _BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
# # IMAGE_DIR        = os.path.normpath(os.path.join(_BASE_DIR, "..", "data", "images"))
# # FAISS_INDEX_PATH = os.path.normpath(os.path.join(_BASE_DIR, "..", "data", "index.faiss"))

# # FACE_MATCH_THRESHOLD  = float(os.environ.get("FACE_MATCH_THRESHOLD",  0.75))
# # FACE_MATCH_NEIGHBORS  = int(os.environ.get("FACE_MATCH_NEIGHBORS",    5))
# # FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# # RECLUSTER_ON_UPLOAD   = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# # RECLUSTER_BATCH_SIZE  = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# # CLIP_SCORE_MIN  = float(os.environ.get("CLIP_SCORE_MIN",  0.14))  # 0.14: catches football/soccer/cat/dog matches
# # THRESHOLD_FLOOR = float(os.environ.get("THRESHOLD_FLOOR", 0.22))
# # ADAPTIVE_RATIO  = float(os.environ.get("ADAPTIVE_RATIO",  0.92))
# # FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.13))  # 0.13: matches CLIP threshold

# # RECLUSTER_COUNTER_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# # RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 30.0))
# # recluster_last_triggered = None

# # COLOR_SCORE_MAP = {
# #     'red':    (1.0, 0,    0),    'blue':   (0,   0,    1.0),
# #     'green':  (0,   1.0,  0),    'yellow': (1.0, 1.0,  0),
# #     'orange': (1.0, 0.5,  0),    'purple': (0.5, 0,    0.5),
# #     'pink':   (1.0, 0.75, 0.8),  'black':  (0,   0,    0),
# #     'white':  (1,   1,    1),    'gray':   (0.5, 0.5,  0.5),
# #     'brown':  (0.6, 0.4,  0.2),
# # }


# # # ── Emotion emojis → search terms (intercept before CLIP) ───────────────────
# # EMOTION_EMOJI_MAP = {
# #     "😊": "happy",  "😄": "happy",  "😁": "happy",  "🙂": "happy",  "😀": "happy",
# #     "😃": "happy",  "🤗": "happy",  "😆": "happy",  "😂": "happy",  "🥰": "happy",
# #     "😍": "happy",  "😎": "happy",  "🥳": "happy",  "😇": "happy",
# #     "😢": "sad",    "😭": "sad",    "😔": "sad",    "😟": "sad",    "🥺": "sad",
# #     "😞": "sad",    "😿": "sad",    "💔": "sad",
# #     "😠": "angry",  "😡": "angry",  "🤬": "angry",  "😤": "angry",  "👿": "angry",
# #     "😲": "surprised", "😮": "surprised", "🤯": "surprised", "😱": "surprised",
# #     "🤢": "disgusted", "🤮": "disgusted", "😷": "disgusted",
# #     "😨": "fearful",   "😰": "fearful",   "😱": "fearful",   "😧": "fearful",
# #     "😐": "neutral",   "😑": "neutral",   "🙄": "neutral",   "😶": "neutral",
# # }

# # # Extended scene emoji map (supplements search_engine.py EMOJI_MAP)
# # SCENE_EMOJI_MAP = {
# #     "🌅": "sunset golden hour", "🌃": "city night",   "🏕️": "camping outdoor",
# #     "🌴": "tropical palm tree", "⛰️": "mountain",     "🌊": "ocean waves beach",
# #     "🎊": "party celebration",  "👪": "family group",  "🤳": "selfie portrait",
# #     "🐕": "dog",                "🐈": "cat",            "🦁": "lion animal",
# #     "🍽️": "food meal",          "☕": "coffee",         "🍰": "cake dessert",
# #     "🏋️": "gym exercise",       "🧘": "yoga meditation","🏊": "swimming pool",
# #     "🌸": "flowers spring",     "🍂": "autumn fall leaves",
# #     "🎸": "music guitar",       "📸": "camera photography",
# #     "⚽": "soccer football sport", "🏀": "basketball sport",
# #     "🏈": "american football sport", "🎾": "tennis sport",
# #     "🏏": "cricket sport bat", "🏒": "hockey sport",
# #     "⚾": "baseball sport",    "🥊": "boxing sport",
# # }

# # # Sport/activity keyword expansions (applied in _clean_query)
# # SPORT_SYNONYMS = {
# #     "football": "football soccer sport player",
# #     "soccer":   "soccer football sport player",
# #     "cricket":  "cricket sport bat player",
# #     "basketball": "basketball sport player",
# #     "tennis":   "tennis sport player",
# #     "swimming": "swimming pool sport",
# #     "running":  "running jogging sport",
# #     "cycling":  "cycling bicycle sport",
# #     "gym":      "gym workout exercise",
# #     "yoga":     "yoga meditation exercise",
# # }

# # # ── Text keyword → emotion mapping ─────────────────────────────────────────
# # EMOTION_KEYWORDS = {
# #     "happy":     ["happy", "happiness", "smiling", "smile", "laughing", "laugh",
# #                   "joy", "joyful", "cheerful", "glad", "delighted", "pleased"],
# #     "sad":       ["sad", "sadness", "crying", "cry", "unhappy", "upset",
# #                   "depressed", "gloomy", "sorrow", "sorrowful", "tears"],
# #     "angry":     ["angry", "anger", "mad", "furious", "rage", "annoyed", "irritated"],
# #     "surprised": ["surprised", "surprise", "shocked", "shock", "amazed", "astonished"],
# #     "fearful":   ["fearful", "fear", "scared", "afraid", "frightened", "terrified"],
# #     "disgusted": ["disgusted", "disgust", "disgusting"],
# #     "neutral":   ["neutral", "expressionless", "blank"],
# # }

# # def _extract_emotion_from_query(query: str):
# #     """Return emotion name if query contains an emotion emoji or keyword, else None."""
# #     # Check emojis first
# #     for emoji, emo in EMOTION_EMOJI_MAP.items():
# #         if emoji in query:
# #             return emo
# #     # Check text keywords
# #     q_lower = query.lower()
# #     for emo, keywords in EMOTION_KEYWORDS.items():
# #         if any(kw in q_lower for kw in keywords):
# #             return emo
# #     return None

# # def _expand_query_emojis(query: str) -> str:
# #     """Replace scene emojis with text equivalents for better CLIP embedding."""
# #     result = query
# #     for emoji, text in SCENE_EMOJI_MAP.items():
# #         result = result.replace(emoji, " " + text + " ")
# #     return " ".join(result.split())


# # def _live(db):
# #     return db.query(DBImage).filter(
# #         (DBImage.is_trashed == False) | (DBImage.is_trashed == None)
# #     )


# # def _img_url(filename: str) -> str:
# #     if not filename:
# #         return None
# #     return os.path.basename(filename)


# # def _user_tags(img) -> list:
# #     """Safely decode the user_tags JSON column into a Python list."""
# #     try:
# #         v = getattr(img, 'user_tags', None)
# #         return json.loads(v) if v and v not in ('[]', '') else []
# #     except Exception:
# #         return []


# # def _clean_query(query: str) -> str:
# #     import unicodedata, re as _re
# #     processed = resolve_query(query)
# #     result = []
# #     for char in processed:
# #         cp = ord(char)
# #         if cp > 127:
# #             cat = unicodedata.category(char)
# #             if cat in ("So", "Sm", "Sk", "Mn"):
# #                 try:
# #                     name = unicodedata.name(char, "").lower()
# #                     name = name.replace(" sign", "").replace(" symbol", "")
# #                     name = _re.sub(r"\bwith\b.*", "", name).strip()
# #                     result.append(" " + name + " ")
# #                 except Exception:
# #                     pass
# #             else:
# #                 result.append(char)
# #         else:
# #             result.append(char)
# #     cleaned = _re.sub(r"\s+", " ", "".join(result)).strip()
# #     return cleaned if cleaned else processed


# # def _startup_fix_filenames():
# #     db = SessionLocal()
# #     try:
# #         fixed = 0
# #         rows = db.query(DBImage).filter(DBImage.filename.contains("/")).all()
# #         for r in rows:
# #             r.filename = os.path.basename(r.filename)
# #             fixed += 1
# #         if fixed:
# #             db.commit()
# #             logger.info(f"Startup fix: normalised {fixed} filenames in DB")
# #     except Exception as e:
# #         db.rollback()
# #         logger.warning(f"Startup filename fix failed: {e}")
# #     finally:
# #         db.close()


# # def _enrich_image(image_id: int, file_path: str):
# #     db = SessionLocal()
# #     try:
# #         img_record = db.query(DBImage).filter(DBImage.id == image_id).first()
# #         if not img_record:
# #             return
# #         try:
# #             img_record.caption_short    = captioning_engine.generate_caption(file_path, max_length=20)
# #             img_record.caption_detailed = captioning_engine.generate_caption(file_path, max_length=60)
# #             vqa_subject = captioning_engine.answer_visual_question(file_path, "What is the main subject in this image?")
# #             vqa_person  = ""
# #             if img_record.person_count and img_record.person_count > 0:
# #                 # Ask BLIP to describe the person — identity recognition
# #                 # works best with very short questions
# #                 vqa_person = captioning_engine.answer_visual_question(
# #                     file_path, "who is this?"
# #                 )
# #                 # Discard generic answers immediately
# #                 if vqa_person and vqa_person.lower().strip() in {
# #                     "man", "woman", "person", "a man", "a woman", "a person",
# #                     "boy", "girl", "child", "human", "unknown", "celebrity",
# #                     "actor", "actress", "no", "yes", "none", "i don't know",
# #                 }:
# #                     vqa_person = ""
# #             img_record.caption_vqa = _json.dumps({
# #                 "subject": vqa_subject,
# #                 "person":  vqa_person,
# #             } if (vqa_subject or vqa_person) else {})
# #             img_record.caption_timestamp = datetime_module.datetime.now()
# #             logger.info(f"✅ Caption [{image_id}]: {img_record.caption_short}")
# #         except Exception as e:
# #             logger.warning(f"⚠️  Caption failed [{image_id}]: {e}")
# #         try:
# #             img_record.ocr_text_enhanced = ocr_engine.extract_text(file_path)
# #             kw = ocr_engine.extract_text_with_confidence(file_path)
# #             img_record.ocr_keywords   = _json.dumps([i["text"] for i in kw if i["confidence"] > 0.7])
# #             img_record.ocr_confidence = float(np.mean([i["confidence"] for i in kw])) if kw else 0.0
# #             logger.info(f"✅ OCR [{image_id}]")
# #         except Exception as e:
# #             logger.warning(f"⚠️  OCR failed [{image_id}]: {e}")
# #         try:
# #             q = image_quality.assess_overall_quality(file_path)
# #             img_record.quality_score = q.get("overall", 0)
# #             img_record.quality_level = q.get("quality_level", "Unknown")
# #             img_record.sharpness     = q.get("sharpness", 0)
# #             img_record.exposure      = q.get("exposure", 0)
# #             img_record.contrast      = q.get("contrast", 0)
# #             img_record.composition   = q.get("composition", 0)
# #             logger.info(f"✅ Quality [{image_id}]: {img_record.quality_level}")
# #         except Exception as e:
# #             logger.warning(f"⚠️  Quality failed [{image_id}]: {e}")
# #         try:
# #             # ── GATE: only run emotion model if InsightFace detected real human faces ──
# #             # InsightFace (ArcFace) is accurate at human faces — won't detect cat faces,
# #             # car headlights, or turtle heads. OpenCV Haar cascade (used by the emotion
# #             # engine) will detect ALL of these as "faces" and classify them wrongly.
# #             # Using the faces table count ensures we only run on real human images.
# #             _face_count_check = db.query(DBFace).filter(DBFace.image_id == image_id).count()
# #             if _face_count_check == 0:
# #                 # No human faces detected by InsightFace → skip emotion model entirely
# #                 img_record.dominant_emotion   = "neutral"
# #                 img_record.face_emotion_count = 0
# #                 img_record.emotion_data       = "[]"
# #                 logger.info(f"⏭️  Emotion [{image_id}]: skipped (no InsightFace faces)")
# #             else:
# #                 ed = emotion_detection.detect_emotions(file_path)
# #                 # ── Confidence filter: only keep high-confidence detections ──────────
# #                 # Low-confidence results (< 0.50) from Haar cascade on non-frontal or
# #                 # partially visible faces produce random emotion labels. Filter them out.
# #                 ed = [e for e in ed if e.get("confidence", 0) >= 0.50]
# #                 img_record.face_emotion_count = len(ed)
# #                 img_record.dominant_emotion   = ed[0]["emotion"] if ed else "neutral"
# #                 img_record.emotion_data       = _json.dumps(ed)
# #                 logger.info(f"✅ Emotion [{image_id}]: {img_record.dominant_emotion} ({len(ed)} faces)")
# #         except Exception as e:
# #             logger.warning(f"⚠️  Emotion failed [{image_id}]: {e}")
# #         try:
# #             a = aesthetic_scoring.score_aesthetics(file_path)
# #             img_record.aesthetic_score  = a.get("aesthetic_score", 0)
# #             img_record.aesthetic_rating = a.get("rating", "Unknown")
# #             logger.info(f"✅ Aesthetic [{image_id}]: {img_record.aesthetic_rating}")
# #         except Exception as e:
# #             logger.warning(f"⚠️  Aesthetic failed [{image_id}]: {e}")
# #         db.commit()
# #         logger.info(f"✅ Enrichment done for image {image_id}")
# #     except Exception as e:
# #         db.rollback()
# #         logger.error(f"❌ Enrichment failed [{image_id}]: {e}", exc_info=True)
# #     finally:
# #         db.close()


# # def should_trigger_recluster(background_tasks):
# #     # FIX: Auto-recluster on upload is DISABLED.
# #     # Recluster wipes all album_id assignments — even with the manual-album fix,
# #     # triggering it automatically on every N uploads causes:
# #     #   (a) uploads to feel slow (recluster runs in same background pool)
# #     #   (b) people section to flicker / re-sort unexpectedly
# #     #   (c) manual albums to potentially get corrupted on edge cases
# #     # Users can click Re-index manually when they want faces/albums refreshed.
# #     pass


# # # ── Lifespan ──────────────────────────────────────────────────────────────────
# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     init_db()
# #     ensure_extra_columns()          # ← runs safely here, after DB is initialised
# #     _startup_fix_filenames()
# #     if os.path.exists(FAISS_INDEX_PATH):
# #         try:
# #             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
# #             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
# #         except Exception as e:
# #             logger.error(f"Index load failed: {e}")
# #             search_engine.index = None
# #     try:
# #         _db = SessionLocal()
# #         try:
# #             faces_total    = _db.query(DBFace).filter(DBFace.face_embedding != None).count()
# #             faces_assigned = _db.query(DBFace).filter(DBFace.face_embedding != None, DBFace.person_id != None).count()
# #             people_total  = _db.query(Person).count()
# #             imgs_in_album = _db.query(DBImage).filter(DBImage.album_id != None).count()
# #             total_imgs    = _db.query(DBImage).count()
# #             needs_face_repair  = faces_total > 0 and (faces_assigned == 0 or people_total == 0)
# #             needs_album_repair = total_imgs  > 0 and imgs_in_album == 0
# #         finally:
# #             _db.close()
# #         if needs_face_repair or needs_album_repair:
# #             logger.warning(f"⚠️  Stale data — faces={faces_total} assigned={faces_assigned} people={people_total} imgs_in_album={imgs_in_album}/{total_imgs}")
# #             logger.info("🔄 Auto-repairing on startup...")
# #             recluster()
# #             logger.info("✅ Auto-repair done")
# #         else:
# #             logger.info(f"✅ DB healthy: {faces_assigned}/{faces_total} faces assigned, {people_total} people, {imgs_in_album}/{total_imgs} imgs in albums")
# #     except Exception as e:
# #         logger.warning(f"Auto-repair check failed (non-fatal): {e}")
# #     logger.info("✅ Ready!")
# #     yield


# # # ── app = FastAPI() MUST be here, before any app.include_router() ─────────────
# # app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)

# # app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# # from starlette.middleware.base import BaseHTTPMiddleware
# # from starlette.requests import Request as StarletteRequest

# # class FixImagePathMiddleware(BaseHTTPMiddleware):
# #     async def dispatch(self, request: StarletteRequest, call_next):
# #         path = request.scope["path"]
# #         if path.startswith("/image/"):
# #             suffix = path[len("/image/"):]
# #             suffix = suffix.lstrip("/")
# #             if suffix.startswith("images/"):
# #                 suffix = suffix[len("images/"):]
# #             bare = os.path.basename(suffix)
# #             if bare:
# #                 request.scope["path"] = f"/image/{bare}"
# #         return await call_next(request)

# # app.add_middleware(FixImagePathMiddleware)

# # if not os.path.exists(IMAGE_DIR):
# #     os.makedirs(IMAGE_DIR)

# # app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # # ── All routers registered AFTER app = FastAPI() ──────────────────────────────
# # app.include_router(dl_router)
# # app.include_router(voice_router)
# # app.include_router(feat_router)    # ← safely here, app already exists


# # # ────────────────────────────────────────────────────────────────────────────
# # # UTILITY
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.get("/health")
# # def health():
# #     return {"status": "ready", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# # @app.get("/debug")
# # def debug_db():
# #     db = SessionLocal()
# #     try:
# #         total_images         = db.query(DBImage).count()
# #         total_faces          = db.query(DBFace).count()
# #         faces_with_embedding = db.query(DBFace).filter(DBFace.face_embedding != None).count()
# #         faces_with_image_id  = db.query(DBFace).filter(DBFace.image_id != None).count()
# #         faces_with_person    = db.query(DBFace).filter(DBFace.person_id != None).count()
# #         total_people         = db.query(Person).count()
# #         total_albums         = db.query(Album).count()
# #         images_in_album      = db.query(DBImage).filter(DBImage.album_id != None).count()
# #         people_detail = []
# #         for p in db.query(Person).all():
# #             faces   = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# #             img_ids = [f.image_id for f in faces if f.image_id]
# #             people_detail.append({"id": p.id, "name": p.name, "face_count": len(faces), "faces_with_image_id": len(img_ids)})
# #         album_detail = []
# #         for a in db.query(Album).all():
# #             imgs = db.query(DBImage).filter(DBImage.album_id == a.id).count()
# #             album_detail.append({"id": a.id, "title": a.title, "image_count": imgs})
# #         return {
# #             "images": total_images, "faces": total_faces,
# #             "faces_with_embedding": faces_with_embedding, "faces_with_image_id": faces_with_image_id,
# #             "faces_with_person": faces_with_person, "people": total_people,
# #             "albums": total_albums, "images_in_album": images_in_album,
# #             "people_detail": people_detail, "album_detail": album_detail,
# #             "action_needed": "Run POST /recluster to fix person/album assignments" if faces_with_person == 0 or images_in_album == 0 else "OK",
# #         }
# #     finally:
# #         db.close()

# # @app.get("/test-db")
# # def test_db():
# #     db = SessionLocal()
# #     try:
# #         count  = db.query(DBImage).count()
# #         images = db.query(DBImage).limit(1).all()
# #         return {
# #             "status": "ok", "total_images": count,
# #             "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None,
# #         }
# #     except Exception as e:
# #         return {"status": "error", "message": str(e)}
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # IMAGE LISTING & SERVING
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.get("/images")
# # def get_all_images(limit: int = 100):
# #     db = SessionLocal()
# #     try:
# #         images = _live(db).limit(limit).all()
# #         return {"results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "face_emotion_count": img.face_emotion_count, "aesthetic_score": img.aesthetic_score, "aesthetic_rating": img.aesthetic_rating, "caption_short": img.caption_short, "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
# #     finally:
# #         db.close()

# # @app.get("/image/{filename:path}")
# # def get_image_file(filename: str):
# #     bare = os.path.basename(filename.lstrip("/"))
# #     if not bare or bare in (".", ".."):
# #         raise HTTPException(status_code=400, detail="Invalid filename")
# #     candidates = [
# #         os.path.join(IMAGE_DIR, bare),
# #         os.path.normpath(os.path.join(os.getcwd(), "..", "data", "images", bare)),
# #         os.path.normpath(os.path.join(os.getcwd(), "data", "images", bare)),
# #         os.path.normpath(os.path.join(os.getcwd(), "images", bare)),
# #     ]
# #     for p in candidates:
# #         if os.path.exists(p):
# #             return FileResponse(p)
# #     db = SessionLocal()
# #     try:
# #         img = db.query(DBImage).filter((DBImage.filename == bare) | (DBImage.filename == f"/images/{bare}")).first()
# #         if img and img.original_path:
# #             p = os.path.normpath(img.original_path.replace("//", "/"))
# #             if os.path.exists(p):
# #                 return FileResponse(p)
# #             alt = os.path.join(IMAGE_DIR, os.path.basename(img.original_path))
# #             if os.path.exists(alt):
# #                 return FileResponse(alt)
# #     finally:
# #         db.close()
# #     logger.warning(f"404 image bare={bare!r} IMAGE_DIR={IMAGE_DIR}")
# #     raise HTTPException(status_code=404, detail=f"Image not found: {bare}")

# # @app.get("/timeline")
# # def get_timeline():
# #     db = SessionLocal()
# #     try:
# #         images = _live(db).order_by(DBImage.timestamp.desc()).all()
# #         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "face_emotion_count": img.face_emotion_count, "aesthetic_score": img.aesthetic_score, "aesthetic_rating": img.aesthetic_rating, "caption_short": img.caption_short, "ocr_text_enhanced": img.ocr_text_enhanced, "person_count": img.person_count, "width": img.width, "height": img.height, "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # DELETE / TRASH / RESTORE
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.post("/delete_image")
# # def delete_image(image_id: int = Form(...)):
# #     db = SessionLocal()
# #     try:
# #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# #         if not img:
# #             raise HTTPException(status_code=404, detail="Image not found")
# #         img.is_trashed = True
# #         img.trashed_at = datetime_module.datetime.now()
# #         db.commit()
# #         return {"success": True, "message": "Image moved to trash"}
# #     finally:
# #         db.close()

# # @app.post("/restore")
# # def restore_image(image_id: int = Form(...)):
# #     db = SessionLocal()
# #     try:
# #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# #         if not img:
# #             raise HTTPException(status_code=404, detail="Image not found")
# #         img.is_trashed = False
# #         img.trashed_at = None
# #         db.commit()
# #         return {"success": True, "message": "Image restored"}
# #     finally:
# #         db.close()

# # @app.post("/permanent_delete")
# # def permanent_delete(image_id: int = Form(...)):
# #     db = SessionLocal()
# #     try:
# #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# #         if not img:
# #             raise HTTPException(status_code=404, detail="Image not found")
# #         filename  = img.filename
# #         file_path = img.original_path or os.path.join(IMAGE_DIR, filename)
# #         bare = os.path.basename(file_path)
# #         for p in [file_path, os.path.join(IMAGE_DIR, bare)]:
# #             if p and os.path.exists(p):
# #                 try:
# #                     os.remove(p)
# #                     break
# #                 except Exception as e:
# #                     logger.warning(f"Could not delete file: {e}")
# #         if search_engine.index is not None:
# #             try:
# #                 # Fast: remove just this one vector — no need to re-embed everything
# #                 ids_arr = np.array([image_id], dtype="int64")
# #                 search_engine.index.remove_ids(ids_arr)
# #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# #                 logger.info(f"✅ FAISS vector {image_id} removed")
# #             except Exception as e:
# #                 logger.warning(f"FAISS remove (non-critical): {e}")
# #         db.query(DBFace).filter(DBFace.image_id == image_id).delete()
# #         db.delete(img)
# #         db.commit()
# #         return {"success": True, "message": f"Image '{filename}' permanently deleted"}
# #     except Exception as e:
# #         db.rollback()
# #         raise HTTPException(status_code=500, detail=str(e))
# #     finally:
# #         db.close()

# # @app.get("/trash")
# # def get_trash():
# #     db = SessionLocal()
# #     try:
# #         images = db.query(DBImage).filter(DBImage.is_trashed == True).all()
# #         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "trashed_at": img.trashed_at.isoformat() if img.trashed_at else None, "timestamp": img.timestamp.isoformat() if img.timestamp else None, "caption": img.caption_short or ""} for img in images]}
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # FAVORITES
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.post("/toggle_favorite")
# # def toggle_favorite(image_id: int = Form(...)):
# #     db = SessionLocal()
# #     try:
# #         img = db.query(DBImage).filter(DBImage.id == image_id).first()
# #         if not img:
# #             raise HTTPException(status_code=404, detail="Image not found")
# #         img.is_favorite = not img.is_favorite
# #         db.commit()
# #         return {"success": True, "is_favorite": img.is_favorite}
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
# #         images = _live(db).filter(DBImage.is_favorite == True).all()
# #         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # SEARCH
# # # ────────────────────────────────────────────────────────────────────────────
# # def _score_candidates(query_emb, sig_words, query_colors, db, top_k, extra_emb=None, text_weight=1.0, image_weight=0.0):
# #     if search_engine.index is None:
# #         return []
# #     if extra_emb is not None:
# #         blended = text_weight * query_emb + image_weight * extra_emb
# #         norm = np.linalg.norm(blended)
# #         if norm > 1e-8:
# #             blended /= norm
# #         q_vec = blended
# #     else:
# #         q_vec = query_emb
# #     total = search_engine.index.ntotal
# #     q = q_vec.reshape(1, -1).astype("float32")
# #     faiss.normalize_L2(q)
# #     distances, indices = search_engine.index.search(q, total)
# #     all_candidates = []
# #     for dist, idx in zip(distances[0], indices[0]):
# #         if idx == -1:
# #             continue
# #         clip_score = float(dist)
# #         if clip_score < CLIP_SCORE_MIN:
# #             break
# #         img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# #         if not img or img.is_trashed:
# #             continue
# #         # (empty-caption guard removed — was killing valid results for colour queries)
# #         # ── OCR bonus (only if most sig_words appear) ──────────────────────
# #         ocr_bonus = 0.0
# #         ocr_src = (img.ocr_text_enhanced or "") + " " + (getattr(img, "ocr_keywords", "") or "")
# #         if sig_words and ocr_src.strip():
# #             txt  = ocr_src.lower()
# #             hits = sum(1 for w in sig_words if w in txt)
# #             # Require at least half the words to match to avoid false boosts
# #             if hits >= max(1, len(sig_words) * 0.5):
# #                 ocr_bonus = min(hits / len(sig_words), 1.0)

# #         # ── Tag bonus ──────────────────────────────────────────────────────
# #         tag_bonus = 0.0
# #         all_tags_text = " ".join(_user_tags(img))
# #         if img.scene_label:
# #             all_tags_text += " " + img.scene_label.lower()
# #         if all_tags_text.strip() and any(w in all_tags_text for w in sig_words):
# #             tag_bonus = 1.0

# #         # ── Caption bonus + contradiction filter ─────────────────────────
# #         caption_bonus = 0.0
# #         hard_exclude    = False  # True = skip this result entirely
# #         caption_bonus   = 0.0
# #         caption_penalty = 0.0
# #         caption_src = " ".join(filter(None, [
# #             img.caption_short or "", img.caption_detailed or ""
# #         ])).lower()

# #         # ── Shared sets (always defined — needed even when sig_words=[]) ──
# #         def _stem(w):
# #             if len(w) > 4 and w.endswith("ses"): return w[:-2]
# #             if len(w) > 3 and w.endswith("ies"): return w[:-3]+"y"
# #             if len(w) > 3 and w.endswith("es"):  return w[:-2]
# #             if len(w) > 3 and w.endswith("s"):   return w[:-1]
# #             return w

# #         _COLOURS = {"black","white","red","blue","green","yellow","orange",
# #                     "purple","pink","brown","gray","grey","golden","silver",
# #                     "nude","beige","cream","navy","teal","maroon","violet"}
# #         _ANIMALS = {
# #             "horse","cow","dog","cat","bird","elephant","tiger","lion",
# #             "bear","wolf","deer","sheep","goat","pig","rabbit","snake",
# #             "fish","chicken","duck","frog","butterfly","spider",
# #             "fox","monkey","giraffe","zebra","panda","kangaroo","crocodile",
# #             "alligator","rhino","hippo","whale","dolphin","shark","eagle",
# #             "parrot","penguin","owl","bat","beetle","ant","bee","crab",
# #             "donkey","camel","llama","raccoon","squirrel","hamster",
# #             "mouse","rat","gecko","lizard","tortoise","turtle","swan",
# #             "otter","seal","walrus","beaver","meerkat","flamingo","peacock",
# #             "cheetah","leopard","jaguar","gorilla","chimpanzee","baboon",
# #             "koala","wombat","hedgehog","ferret","chinchilla","iguana",
# #             "chameleon","scorpion","jellyfish","octopus","lobster","shrimp",
# #             "kitten","puppy","calf","foal","piglet","lamb","cub","chick",
# #         }
# #         _CLOTHING = {"dress","suit","shirt","jacket","sari","skirt","gown",
# #                      "uniform","coat","blouse","top","hoodie","sweater",
# #                      "saree","kurta","lehenga","tuxedo","blazer","vest",
# #                      "bikini","swimsuit","robe","kimono","pyjama","shorts",
# #                      "trouser","pant","jeans","legging"}
# #         _GENDER = {"woman","girl","lady","female","man","boy","male","guy"}
# #         _FEMALE = {"woman","girl","lady","female"}
# #         _MALE   = {"man","boy","guy","male"}
# #         _HUMAN_WORDS = {"man","woman","boy","girl","person","people","lady",
# #                         "guy","child","adult","player","actor","actress"}

# #         # Always compute q_* from sig_words (empty set when sig_words=[])
# #         _q_set     = set(sig_words)
# #         q_colours  = _COLOURS  & _q_set
# #         q_animals  = _ANIMALS  & _q_set
# #         q_clothing = _CLOTHING & _q_set
# #         q_gender   = _GENDER   & _q_set

# #         # Always compute cap_words from caption
# #         if caption_src:
# #             _cap_raw  = caption_src.split()
# #             cap_words = {_stem(w) for w in _cap_raw} | set(_cap_raw)
# #         else:
# #             cap_words = set()

# #         if sig_words and caption_src:
# #             # ── Caption bonus ────────────────────────────────────────────
# #             hits = sum(1 for w in _q_set if w in caption_src)
# #             match_ratio = hits / len(_q_set)
# #             if match_ratio >= 0.5:
# #                 caption_bonus = match_ratio

# #             # ── 1. Animal mismatch ───────────────────────────────────────
# #             if q_animals:
# #                 cap_animals = _ANIMALS & cap_words
# #                 if cap_animals and not (cap_animals & q_animals):
# #                     hard_exclude = True

# #             # ── 2. Colour + clothing contradiction (context-aware) ──────
# #             if not hard_exclude and q_colours:
# #                 cap_colours  = _COLOURS & cap_words
# #                 cap_clothing = _CLOTHING & cap_words
# #                 same_cloth   = bool(cap_clothing & q_clothing)
# #                 contradicting = cap_colours - q_colours
# #                 matching      = cap_colours & q_colours
# #                 if same_cloth and cap_colours:
# #                     # Caption has same clothing item → colour must match
# #                     # e.g. "white dress" for "black dress" → EXCLUDED
# #                     # "black hair wearing white dress" → "black" matches query but 
# #                     # the DRESS colour (white) contradicts → still exclude
# #                     if contradicting and not matching:
# #                         hard_exclude = True
# #                     elif contradicting and matching:
# #                         caption_penalty += 0.10
# #                 elif not same_cloth and cap_colours and contradicting and not matching:
# #                     caption_penalty += 0.18

# #                         # ── 3. Clothing constraint — hard exclude on mismatch ────────
# #             if not hard_exclude and q_clothing:
# #                 cap_clothing = _CLOTHING & cap_words
# #                 if cap_clothing and not (cap_clothing & q_clothing):
# #                     # Caption names a DIFFERENT clothing item (e.g. "suit" for "dress")
# #                     hard_exclude = True
# #                 elif not cap_clothing:
# #                     # Caption has NO clothing word at all
# #                     if q_colours and (q_colours & cap_words):
# #                         # Colour word appears in caption (e.g. "red eyes") but no clothing
# #                         # → the colour belongs to something else, not the dress → exclude
# #                         hard_exclude = True
# #                     elif q_colours and not (q_colours & cap_words):
# #                         # No clothing AND wrong/no colour → definitely not the right image
# #                         hard_exclude = True

# #             # ── 4. Gender mismatch ───────────────────────────────────────
# #             if not hard_exclude and q_gender:
# #                 q_female = bool(_FEMALE & q_gender)
# #                 q_male   = bool(_MALE   & q_gender)
# #                 if q_female and (_MALE & cap_words) and not (_FEMALE & cap_words):
# #                     hard_exclude = True   # query=woman but caption has only men → always exclude
# #                 if q_male and (_FEMALE & cap_words) and not (_MALE & cap_words):
# #                     hard_exclude = True   # query=man but caption has only women → always exclude

# #             if caption_penalty >= 0.40:
# #                 hard_exclude = True
# #             caption_penalty = min(caption_penalty, 0.38)

# #         # ── Animal query: exclude human-only captions (always runs) ──────
# #         if not hard_exclude and q_animals and not q_gender and caption_src:
# #             cap_has_human  = bool(_HUMAN_WORDS & cap_words)
# #             cap_has_animal = bool(_ANIMALS & cap_words)
# #             if cap_has_human and not cap_has_animal:
# #                 hard_exclude = True

# #         # ── BUGFIX: Human query: exclude animal-only captions ───────────
# #         if not hard_exclude and (_HUMAN_WORDS & _q_set) and caption_src:
# #             cap_has_human  = bool(_HUMAN_WORDS & cap_words)
# #             cap_has_animal = bool(_ANIMALS & cap_words)
# #             if cap_has_animal and not cap_has_human:
# #                 hard_exclude = True

# #         # ── BUGFIX: Empty/missing caption — use scene_label as fallback ─
# #         # When caption_src is empty (BLIP not yet run or failed), ALL the
# #         # animal/human filters above are skipped because they check
# #         # `if sig_words and caption_src`.  Non-matching animals then leak
# #         # through with a pure CLIP score.  Use scene_label + user_tags to
# #         # catch the worst offenders.
# #         if not hard_exclude and not caption_src and q_animals:
# #             # Build a stem-aware token set from scene_label + user_tags
# #             _sl = (img.scene_label or "").lower()
# #             _ut = " ".join(_user_tags(img)).lower()
# #             _fb_raw = set((_sl + " " + _ut).split())
# #             _fb_toks = _fb_raw | {_stem(w) for w in _fb_raw}
# #             _fb_animals = _ANIMALS & _fb_toks
# #             # If scene_label names a DIFFERENT animal from the query → exclude
# #             if _fb_animals and not (_fb_animals & q_animals):
# #                 hard_exclude = True
# #             # If scene_label has NO animal at all but has human words → exclude
# #             # (e.g. "person, sink" scene_label for a cat query)
# #             if not _fb_animals:
# #                 _fb_humans = _HUMAN_WORDS & _fb_toks
# #                 if _fb_humans:
# #                     hard_exclude = True

# #         # ── BUGFIX: Specific object words absent from caption ────────────────
# #         # If query mentions a concrete specific object (flower, rose, guitar, etc.)
# #         # and the caption exists but doesn't mention it at all → CLIP is hallucinating
# #         # semantic similarity. These concrete words almost never produce false negatives
# #         # (if the image has a flower, BLIP will say "flower").
# #         _SPECIFIC_OBJECTS = {
# #             "flower","rose","tulip","sunflower","daisy","bouquet","petal",
# #             "guitar","piano","violin","drum","trumpet","saxophone",
# #             "cake","pizza","burger","sandwich","coffee","wine","beer","cocktail",
# #             "trophy","medal","cup","award","crown",
# #             "balloon","umbrella","candle","lantern","flag",
# #             "sword","shield","arrow","bow",
# #             "basketball","football","soccer","tennis","cricket","baseball",
# #             "bicycle","motorcycle","skateboard","surfboard","snowboard",
# #         }
# #         if not hard_exclude and caption_src and sig_words:
# #             _q_specific = _SPECIFIC_OBJECTS & set(sig_words)
# #             if _q_specific:
# #                 # Any of the specific objects must appear in caption (stemmed)
# #                 _q_specific_stemmed = {_stem(w) for w in _q_specific} | _q_specific
# #                 if not (_q_specific_stemmed & cap_words):
# #                     hard_exclude = True

# #         # ── Hard exclude — skip entirely ────────────────────────────────
# #         if hard_exclude:
# #             continue

# #         # ── Color bonus ──────────────────────────────────────────────────
# #         color_bonus = 0.0
# #         if query_colors and getattr(img, "avg_r", None) is not None:
# #             img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
# #             for qc in query_colors:
# #                 d = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
# #                 color_bonus = max(color_bonus, max(0.0, 1.0 - d / np.sqrt(3)))

# #         # ── Final score ──────────────────────────────────────────────────
# #         exact_match_bonus = 0.0
# #         if sig_words and caption_src:
# #             all_present = all(w in caption_src for w in sig_words)
# #             majority_present = sum(1 for w in sig_words if w in caption_src) / len(sig_words) >= 0.7
# #             if all_present:
# #                 exact_match_bonus = 0.25
# #             elif majority_present:
# #                 exact_match_bonus = 0.12

# #         # Dynamic colour weight: when query explicitly asks for a colour
# #         # (e.g. "white horse", "red car"), pixel colour scoring should matter
# #         # much more than the default 0.01. Raise it to 0.15 so a brown horse
# #         # scores significantly lower than a white horse for "white horse".
# #         _colour_weight = 0.15 if query_colors else 0.01

# #         final_score = (0.60 * clip_score
# #                       + 0.20 * caption_bonus
# #                       + 0.15 * exact_match_bonus
# #                       + 0.03 * tag_bonus
# #                       + 0.01 * ocr_bonus
# #                       + _colour_weight * color_bonus
# #                       - caption_penalty)
# #         all_candidates.append({"img": img, "clip": clip_score, "final": final_score})
# #     all_candidates.sort(key=lambda x: x["final"], reverse=True)
# #     kept = [c for c in all_candidates if c["final"] >= FINAL_SCORE_MIN]
# #     if all_candidates:
# #         logger.info(f"📊 CLIP: {len(all_candidates)} above CLIP_MIN, {len(kept)} above FINAL_MIN={FINAL_SCORE_MIN}, top={all_candidates[0]['final']:.3f} bottom={all_candidates[-1]['final']:.3f}")
# #     results = []
# #     for c in kept[:top_k]:
# #         img = c["img"]
# #         results.append({
# #                 "id": img.id,
# #                 "filename": _img_url(img.filename),
# #                 "score": round(c["final"] * 100, 2),
# #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# #                 "person_count": img.person_count or 0,
# #                 "caption_short": img.caption_short or "",
# #                 "caption_detailed": img.caption_detailed or "",
# #                 "quality_score": img.quality_score or 0,
# #                 "quality_level": img.quality_level or "",
# #                 "sharpness": img.sharpness or 0,
# #                 "exposure": img.exposure or 0,
# #                 "contrast": img.contrast or 0,
# #                 "composition": img.composition or 0,
# #                 "dominant_emotion": img.dominant_emotion or "",
# #                 "face_emotion_count": img.face_emotion_count or 0,
# #                 "aesthetic_score": img.aesthetic_score or 0,
# #                 "aesthetic_rating": img.aesthetic_rating or "",
# #                 "ocr_text_enhanced": img.ocr_text_enhanced or "",
# #                 "scene_label": img.scene_label or "",
# #                 "width": img.width,
# #                 "height": img.height,
# #                 "is_favorite": bool(img.is_favorite),
# #                 "user_tags": _user_tags(img),
# #                 "photo_note": getattr(img, "photo_note", "") or "",
# #             })
# #     return results

# # @app.post("/search")
# # def search(query: str = Form(...), top_k: int = Form(20)):
# #     if not query or not query.strip():
# #         return {"status": "error", "message": "Query empty"}

# #     # ── Emotion search: emoji OR text keywords → filter by dominant_emotion ──
# #     detected_emotion = _extract_emotion_from_query(query)
# #     is_pure_emoji    = query.strip() in EMOTION_EMOJI_MAP

# #     # Check if query is MOSTLY about emotion (emotion word + optional "faces"/"photos" etc)
# #     EMOTION_MODIFIERS = {"faces", "face", "photos", "photo", "pictures", "picture",
# #                          "images", "image", "people", "person", "moments", "looking",
# #                          "expression", "expressions", "ones", "me", "us", "all",
# #                          # BUGFIX: common query prefixes that were left in non_emotion_words
# #                          # causing "show me happy faces" / "find sad photos" to miss the
# #                          # emotion route and fall through to CLIP which returns nothing.
# #                          "show", "find", "get", "give", "display", "share",
# #                          "my", "the", "a", "with", "some", "any"}
# #     query_words = set(query.lower().split())
# #     non_emotion_words = query_words - EMOTION_MODIFIERS -         {kw for kws in EMOTION_KEYWORDS.values() for kw in kws}
# #     is_emotion_query = detected_emotion and len(non_emotion_words) == 0

# #     if detected_emotion and (is_pure_emoji or is_emotion_query):
# #         # Route directly to emotion DB filter
# #         # Only match images where emotion was detected by face analysis
# #         # (face_emotion_count > 0 means actual faces were detected and analysed)
# #         db = SessionLocal()
# #         try:
# #             # Require: actual faces detected (face_emotion_count>0) AND human present (person_count>0)
# #             # This prevents turtles/cars/cats from appearing in happy/angry emotion search
# #             # Strict: only images where faces were actually detected + analysed
# #             # face_emotion_count > 0 = real face detection ran on this image
# #             imgs = _live(db).filter(
# #                 DBImage.dominant_emotion == detected_emotion,
# #                 DBImage.face_emotion_count > 0,
# #             ).order_by(DBImage.timestamp.desc()).limit(top_k).all()
# #             # NO fallback that removes face_emotion_count requirement.
# #             # Without this guard, cars/turtles/spider-man with wrongly assigned
# #             # emotion labels appear in results. Better to show fewer real results.
# #             if imgs:
# #                 return {"status": "found", "query": query, "count": len(imgs),
# #                         "results": [{"id": img.id, "filename": _img_url(img.filename),
# #                                      "score": 90.0,
# #                                      "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                                      "caption_short": img.caption_short or "",
# #                                      "user_tags": _user_tags(img),
# #                                      "dominant_emotion": img.dominant_emotion,
# #                                      "person_count": img.person_count or 0,
# #                                      "photo_note": getattr(img,"photo_note","") or ""} for img in imgs]}
# #             else:
# #                 # No results — emotions may not be detected yet
# #                 return {"status": "not_found",
# #                         "message": f"No photos with '{detected_emotion}' emotion found. "
# #                                    f"Click 'Fix Emotions' in the sidebar to detect emotions on your photos."}
# #         finally:
# #             db.close()

# #     # ── Expand scene emojis to text for CLIP ────────────────────────────────
# #     expanded_query = _expand_query_emojis(query)
# #     # Expand sport/activity synonyms for better CLIP recall
# #     _words = expanded_query.lower().split()
# #     if any(w in SPORT_SYNONYMS for w in _words):
# #         _words_exp = []
# #         for w in _words:
# #             _words_exp.append(SPORT_SYNONYMS.get(w, w))
# #         expanded_query = " ".join(_words_exp)
# #     processed_query = _clean_query(expanded_query)
# #     logger.info(f"🔍 Search: '{query}' → '{processed_query}'")

# #     # BUGFIX: q_tokens_pre must be defined HERE — it is used by the
# #     # should_search_people pre-computation block below, which runs before
# #     # the later assignment at the CLIP section (line ~17609).
# #     q_tokens_pre = query.lower().strip().split()

# #     # If emotion emoji mixed with other words, boost emotion filter
# #     emotion_boost = detected_emotion  # may be None

# #     # ── Pre-compute should_search_people so sig_words can use it ────────────
# #     # (Full person-search logic runs later; here we just need the flag)
# #     _DESCRIPTOR_PRE = {
# #         "a","an","the","in","on","at","of","for","with","by","to","from","into",
# #         "is","are","was","were","be","been","has","have","had","do","does","did",
# #         "will","would","could","should","may","might","can","wearing","holding",
# #         "standing","sitting","posing","looking","long","short","tall","small",
# #         "big","large","little","young","old","beautiful","pretty","black","white",
# #         "red","blue","green","yellow","orange","purple","pink","brown","gray",
# #         "woman","man","girl","boy","lady","guy","person","people","female","male",
# #         "human","child","adult","baby","player","actor","photo","image","picture",
# #         "standing","next","front","back","window","field","camera","dress","suit",
# #         "shirt","jacket","hair","smiling",
# #         # FIX: query-prefix words that aren't person names
# #         "show","find","get","give","display","search","look","me","my","us","our",
# #         # FIX: emotion/state adjectives that aren't person names
# #         "happy","sad","angry","surprised","fearful","neutral","disgusted",
# #         "smiling","laughing","crying","excited","bored","tired","scared",
# #         # FIX: common descriptive adjectives that confused the name detector
# #         "funny","cute","cool","nice","great","good","bad","best","worst",
# #         "images","photos","pictures","moments","all","some","any","many",
# #     }
# #     # Animal/object words that look like names but are NOT people
# #     _NON_PERSON_WORDS = {
# #         # animals
# #         "cat","dog","horse","cow","bird","fish","fox","lion","tiger","bear","wolf",
# #         "pig","rabbit","duck","frog","snake","deer","sheep","goat","chicken","otter",
# #         "seal","panda","koala","monkey","giraffe","zebra","elephant","whale","shark",
# #         "eagle","owl","penguin","parrot","bee","ant","spider","butterfly","kitten",
# #         "puppy",
# #         # nature / places
# #         "sunset","beach","ocean","mountain","forest","river","flower","tree",
# #         "sky","snow","rain","night","city","park","road","bridge","building","house",
# #         # vehicles / objects
# #         "car","bus","train","plane","boat","food","pizza","cake","coffee",
# #         # fictional characters & universes — prevent person-search triggering
# #         "ironman","spiderman","batman","superman","thor","hulk","avengers",
# #         "marvel","anime","cartoon","fictional","hero","villain","superhero",
# #         "comic","spider","thanos","loki","deadpool","wolverine","aquaman",
# #         "flash","wonder","hawkeye","ultron","venom","groot","rocket","nebula",
# #         "gamora","starlord","antman","blackwidow","scarlet","vision","warmachine",
# #         "falcon","panther","shuri","okoye","wakanda","asgard","xmen","cyclops",
# #         "magneto","mystique","beast","nightcrawler","storm","phoenix","rogue",
# #         # music genres / scene words (not person names)
# #         "kpop","kdrama","jpop","bollywood","hollywood","netflix","disney","pixar",
# #         "hiphop","hiphop","jazz","rock","pop","metal","indie","classical",
# #         # common OCR / UI words that appear as image text
# #         "user","login","password","email","search","home","menu","settings",
# #         "button","click","submit","cancel","back","next","done","ok","yes","no",
# #         "error","loading","please","enter","select","upload","download","share",
# #         "follow","like","comment","post","profile","account","app","website",
# #         # generic concept words
# #         "vintage","retro","modern","classic","dark","light","minimal","abstract",
# #         "nature","urban","street","indoor","outdoor","close","wide","macro",
# #     }
# #     _min_len_pre      = 3 if len(q_tokens_pre) <= 3 else 4
# #     _nc_pre           = [w for w in q_tokens_pre if len(w) >= _min_len_pre
# #                          and w not in _DESCRIPTOR_PRE
# #                          and w not in _NON_PERSON_WORDS
# #                          and w.isalpha()]
# #     _dc_pre           = sum(1 for w in q_tokens_pre if w in _DESCRIPTOR_PRE)
# #     _is_desc_pre      = _dc_pre >= len(q_tokens_pre) * 0.5
# #     should_search_people = bool(_nc_pre) and not _is_desc_pre


# #     # ── Tag fast-path ─────────────────────────────────────────────────────────
# #     # BUGFIX: Arbitrary user tags (e.g. "birthday", "vacation", "grandma") only
# #     # contributed a 0.03 weight bonus to CLIP scores, making them essentially
# #     # invisible when typed in the search bar. Now: if ANY word in the query
# #     # exactly matches a known user tag in the DB, we return those tagged images
# #     # immediately (score=97) merged with any further CLIP/OD results.
# #     # This runs BEFORE the OD and CLIP paths so tags always take priority.
# #     _q_words_lower   = set(query.lower().strip().split())
# #     _tag_fast_results = []
# #     _tag_fast_ids     = set()
# #     try:
# #         _db_tag = SessionLocal()
# #         try:
# #             # Collect all distinct tags that exist in the DB
# #             _all_tag_rows = _db_tag.execute(text(
# #                 "SELECT DISTINCT user_tags FROM images "
# #                 "WHERE user_tags IS NOT NULL AND user_tags != '[]' "
# #                 "AND (is_trashed IS NULL OR is_trashed=0)"
# #             )).fetchall()
# #             import json as _json_tag
# #             _known_tags = set()
# #             for _trow in _all_tag_rows:
# #                 try:
# #                     for _t in _json_tag.loads(_trow[0] or "[]"):
# #                         _known_tags.add(_t.strip().lower())
# #                 except Exception:
# #                     pass
# #             # Find query words that are known tags
# #             _matched_tags = _q_words_lower & _known_tags
# #             if _matched_tags:
# #                 for _img in _live(_db_tag).all():
# #                     _itags = {t.strip().lower() for t in _user_tags(_img)}
# #                     if _itags & _matched_tags:
# #                         if _img.id not in _tag_fast_ids:
# #                             _tag_fast_ids.add(_img.id)
# #                             _tag_fast_results.append({
# #                                 "id": _img.id,
# #                                 "filename": _img_url(_img.filename) or "",
# #                                 "score": 97.0,
# #                                 "timestamp": _img.timestamp.isoformat() if _img.timestamp else None,
# #                                 "caption_short": _img.caption_short or "",
# #                                 "person_count": _img.person_count or 0,
# #                                 "dominant_emotion": _img.dominant_emotion or "",
# #                                 "quality_level": _img.quality_level or "",
# #                                 "quality_score": _img.quality_score or 0,
# #                                 "aesthetic_score": _img.aesthetic_score or 0,
# #                                 "user_tags": _user_tags(_img),
# #                                 "photo_note": getattr(_img, "photo_note", "") or "",
# #                                 "is_favorite": bool(_img.is_favorite),
# #                                 "scene_label": _img.scene_label or "",
# #                                 "width": _img.width, "height": _img.height,
# #                             })
# #                 logger.info(f"🏷️ Tag fast-path: matched tags {_matched_tags} → {len(_tag_fast_results)} images")
# #         finally:
# #             _db_tag.close()
# #     except Exception as _tag_err:
# #         logger.warning(f"Tag fast-path failed (non-fatal): {_tag_err}")

# #     # ── Object-detection label fast-path ──────────────────────────────────
# #     # Check if any query word directly maps to a COCO-detected object.
# #     # scene_label contains comma-separated Faster R-CNN detections (e.g. "cat, person").
# #     # This is far more reliable than CLIP for specific animals/objects.
# #     QUERY_TO_COCO = {
# #         "cat":"cat","cats":"cat","kitten":"cat","kitty":"cat","kittens":"cat",
# #         "dog":"dog","dogs":"dog","puppy":"dog","puppies":"dog","pup":"dog",
# #         "horse":"horse","horses":"horse","pony":"horse",
# #         "bird":"bird","birds":"bird","parrot":"bird","eagle":"bird","owl":"bird",
# #         "cow":"cow","cows":"cow","bull":"cow","cattle":"cow",
# #         "elephant":"elephant","bear":"bear","zebra":"zebra","giraffe":"giraffe",
# #         "sheep":"sheep","goat":"sheep","deer":"deer","rabbit":"rabbit",
# #         "soccer":"sports ball","football":"sports ball","sport":"sports ball",
# #         "cricket":"baseball bat","baseball":"baseball bat",
# #         "tennis":"tennis racket","racket":"tennis racket",
# #         "surfing":"surfboard","surf":"surfboard","skateboard":"skateboard",
# #         "skiing":"skis","ski":"skis","snowboard":"snowboard",
# #         "car":"car","cars":"car","truck":"truck","bus":"bus",
# #         "motorcycle":"motorcycle","bicycle":"bicycle","bike":"bicycle",
# #         "boat":"boat","ship":"boat","train":"train",
# #         "airplane":"airplane","plane":"airplane","aircraft":"airplane",
# #         "pizza":"pizza","cake":"cake","sandwich":"sandwich",
# #         "apple":"apple","banana":"banana","orange":"orange",
# #         "bottle":"bottle","cup":"cup","wine":"wine glass",
# #         "laptop":"laptop","keyboard":"keyboard","mouse":"mouse",
# #         "phone":"cell phone","cellphone":"cell phone",
# #         "chair":"chair","sofa":"couch","couch":"couch","bed":"bed",
# #         "tv":"tv","television":"tv","monitor":"tv",
# #         "book":"book","books":"book","clock":"clock","vase":"vase",
# #         "scissors":"scissors","knife":"knife","fork":"fork","spoon":"spoon",
# #     }
# #     _q_lower_words = set(query.lower().split())
# #     _coco_targets = set()
# #     for _w in _q_lower_words:
# #         if _w in QUERY_TO_COCO:
# #             _coco_targets.add(QUERY_TO_COCO[_w])
    
# #     if _coco_targets and not should_search_people:
# #         # Direct object detection search — bypass CLIP entirely for these queries
# #         _db_od = SessionLocal()
# #         try:
# #             _od_results = []
# #             _seen_od    = set()   # dedup by image id
# #             _seen_fn    = set()   # dedup by filename (catches re-indexed duplicates)

# #             # Broad animal vocabulary used by BLIP captions — used below to
# #             # verify that the COCO model's scene_label isn't a misclassification.
# #             _BROAD_ANIMALS = {
# #                 "horse","cow","dog","cat","bird","elephant","tiger","lion","bear",
# #                 "wolf","deer","sheep","goat","pig","rabbit","snake","fish","chicken",
# #                 "duck","frog","fox","monkey","giraffe","zebra","panda","otter",
# #                 "squirrel","raccoon","hamster","mouse","rat","kitten","puppy","calf",
# #                 "foal","piglet","lamb","cub","chick","parrot","penguin","owl","bat",
# #                 "seal","beaver","meerkat","flamingo","peacock","cheetah","leopard",
# #                 "jaguar","gorilla","koala","hedgehog","iguana","chameleon","scorpion",
# #                 "jellyfish","octopus","lobster","crab","shrimp","bison","buffalo",
# #                 "moose","elk","reindeer","caribou","hyena","wildebeest","gazelle",
# #             }

# #             def _od_row(img, score):
# #                 return {
# #                     "id": img.id,
# #                     "filename": _img_url(img.filename),
# #                     "score": score,
# #                     "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                     "caption_short": img.caption_short or "",
# #                     "person_count": img.person_count or 0,
# #                     "dominant_emotion": img.dominant_emotion or "",
# #                     "quality_level": img.quality_level or "",
# #                     "quality_score": img.quality_score or 0,
# #                     "aesthetic_score": img.aesthetic_score or 0,
# #                     "user_tags": _user_tags(img),
# #                     "photo_note": getattr(img, "photo_note", "") or "",
# #                     "is_favorite": bool(img.is_favorite),
# #                 }

# #             for _img in _live(_db_od).all():
# #                 fn = _img_url(_img.filename) or ""
# #                 # ── BUGFIX: dedup by filename to catch re-indexed duplicates ──
# #                 if _img.id in _seen_od or fn in _seen_fn:
# #                     continue

# #                 # ── BUGFIX: also match via user_tags (manual tags bypass OD) ──
# #                 _img_tags = {t.strip().lower() for t in _user_tags(_img)}
# #                 is_tag_match = bool(_img_tags & _coco_targets)

# #                 # ── scene_label match (original OD path) ─────────────────────
# #                 is_od_match = False
# #                 _match_score = 0
# #                 if _img.scene_label:
# #                     _detected = {x.strip().lower() for x in _img.scene_label.split(",")}
# #                     _match_score = sum(1 for t in _coco_targets if t in _detected)
# #                     is_od_match = _match_score > 0

# #                 if not is_od_match and not is_tag_match:
# #                     continue

# #                 # ── BUGFIX: caption-verify OD matches to filter COCO misclassifications ──
# #                 # Faster R-CNN doesn't know fox/otter/squirrel/rabbit/panda (not COCO
# #                 # classes). It often mislabels them as "cat" or "dog". BLIP captions
# #                 # are far more accurate — if the caption names a DIFFERENT animal,
# #                 # the scene_label detection is a false positive. Skip it.
# #                 # Exception: user-tagged images are always trusted.
# #                 if is_od_match and not is_tag_match:
# #                     _cap = (_img.caption_short or "").lower()
# #                     if _cap:
# #                         _cap_toks_raw = set(_cap.split())
# #                         def _stem_tok(w):
# #                             if len(w) > 4 and w.endswith("ses"): return w[:-2]
# #                             if len(w) > 4 and w.endswith("ies"): return w[:-3]+"y"
# #                             if len(w) > 3 and w.endswith("es"):  return w[:-2]
# #                             if len(w) > 3 and w.endswith("s"):   return w[:-1]
# #                             return w
# #                         _cap_toks = _cap_toks_raw | {_stem_tok(w) for w in _cap_toks_raw}
# #                         _cap_animals = _BROAD_ANIMALS & _cap_toks
# #                         if _cap_animals and not (_cap_animals & _coco_targets):
# #                             logger.info(
# #                                 f"🚫 OD-filter: id={_img.id} caption has {_cap_animals}, "
# #                                 f"not {_coco_targets} — skipping misclassification"
# #                             )
# #                             continue
# #                     else:
# #                         # Caption empty/missing — fall back to scene_label cross-check.
# #                         # If scene_label contains ONLY human labels and no animal,
# #                         # the "cat"/"dog" detection is almost certainly a false positive.
# #                         _sl_raw = (_img.scene_label or "").lower()
# #                         _sl_toks = set(_sl_raw.replace(",", " ").split())
# #                         _sl_animals = _BROAD_ANIMALS & _sl_toks
# #                         _HUMAN_SL   = {"person", "man", "woman", "boy", "girl", "people"}
# #                         _sl_humans  = _HUMAN_SL & _sl_toks
# #                         if _sl_humans and not _sl_animals:
# #                             logger.info(
# #                                 f"🚫 OD-filter (no caption): id={_img.id} scene_label has "
# #                                 f"only humans {_sl_humans}, not {_coco_targets} — skipping"
# #                             )
# #                             continue

# #                 # FIX Bug2: caption exists, no animals detected, but query IS for an animal
# #                 # e.g. "two men in suits" for a "horse" query → exclude
# #                 if is_od_match and not is_tag_match and _cap:
# #                     _cap_toks_check = {w.rstrip("s") if len(w) > 3 else w
# #                                        for w in set(_cap.split())} | set(_cap.split())
# #                     _cap_animals_check = _BROAD_ANIMALS & _cap_toks_check
# #                     _animal_targets = _coco_targets & {
# #                         "cat","dog","horse","bird","cow","elephant","bear",
# #                         "zebra","giraffe","sheep","deer","rabbit","horse",
# #                     }
# #                     if _animal_targets and not _cap_animals_check:
# #                         # Caption has NO animal — check if it's a human-only image
# #                         _HUMAN_CAP = {"man","woman","person","girl","boy","people","men",
# #                                       "women","actor","actress","player","character","he",
# #                                       "she","they","guy","lady","child","adult"}
# #                         if _HUMAN_CAP & _cap_toks_check:
# #                             logger.info(
# #                                 f"🚫 OD-filter (human-only caption): id={_img.id} "
# #                                 f"caption '{_cap[:60]}' has no animal for {_animal_targets}"
# #                             )
# #                             continue

# #                 # FIX Bug4: non-animal OD targets (car, truck, etc.) with unrelated captions
# #                 # e.g. "thor and thor in the avengers movie" for a "red car" query → exclude
# #                 if is_od_match and not is_tag_match and _cap:
# #                     _VEHICLE_COCO = {"car","truck","bus","motorcycle","bicycle","boat",
# #                                      "airplane","train"}
# #                     _vehicle_targets = _coco_targets & _VEHICLE_COCO
# #                     if _vehicle_targets:
# #                         _VEHICLE_WORDS = {"car","truck","bus","motorcycle","bike","bicycle",
# #                                           "vehicle","driving","road","highway","parking",
# #                                           "train","boat","ship","airplane","plane","suv",
# #                                           "sedan","sports","jeep","porsche","ferrari",
# #                                           "bmw","mercedes","driving","auto","automobile"}
# #                         _FICTIONAL_WORDS = {"marvel","avengers","thor","ironman","iron",
# #                                             "spiderman","batman","superman","hulk","anime",
# #                                             "cartoon","movie","film","fictional","hero",
# #                                             "villain","superhero","comic","bust","armor",
# #                                             "avenger","character","suit"}
# #                         _cap_set_v = set(_cap.split())
# #                         _has_vehicle_word = bool(_VEHICLE_WORDS & _cap_set_v)
# #                         _has_fictional    = bool(_FICTIONAL_WORDS & _cap_set_v)
# #                         if not _has_vehicle_word and _has_fictional:
# #                             logger.info(
# #                                 f"🚫 OD-filter (fictional caption for vehicle): id={_img.id} "
# #                                 f"caption '{_cap[:60]}'"
# #                             )
# #                             continue

# #                 score = 92.0 if is_tag_match else round(85.0 + _match_score * 5, 1)

# #                 # ── BUGFIX: Colour-word filter in OD path ─────────────────────
# #                 # The OD fast-path matches by object class (horse/cat/car) but
# #                 # completely ignores colour words in the query.
# #                 # "white horse" was returning brown/dark horses because OD just
# #                 # found all horses and returned them all at score 85–90.
# #                 #
# #                 # Rules (only when query contains colour words):
# #                 #   1. Caption has colour that CONTRADICTS query → EXCLUDE
# #                 #      e.g. "brown horse" for "white horse" → skip
# #                 #   2. Caption has colour that MATCHES query → KEEP (boosted)
# #                 #   3. Caption exists but has NO colour word:
# #                 #      → Don't return from OD; fall through to CLIP which uses
# #                 #        pixel-level colour scoring (avg_r/g/b) — much more reliable
# #                 #        for unlabelled subjects
# #                 #   4. No caption yet (BLIP not run) → KEEP (can't verify)
# #                 if not is_tag_match and _q_lower_words:
# #                     _COLOUR_WORDS = {
# #                         "white","black","red","blue","green","yellow","orange",
# #                         "purple","pink","brown","gray","grey","golden","silver",
# #                         "dark","light","pale","cream","beige","spotted","striped",
# #                     }
# #                     _q_colours = _COLOUR_WORDS & _q_lower_words
# #                     if _q_colours:
# #                         _cap_for_colour = (_img.caption_short or "").lower()
# #                         if _cap_for_colour:  # only apply filter if caption exists
# #                             _cap_toks_c = set(_cap_for_colour.split())
# #                             _cap_colours = _COLOUR_WORDS & _cap_toks_c
# #                             _matching     = _q_colours & _cap_colours
# #                             _contradicting = _cap_colours - _q_colours

# #                             # "black and white photo/photograph" = monochrome style,
# #                             # not a colour description of the subject → don't exclude
# #                             _is_mono = (("black" in _cap_toks_c or "white" in _cap_toks_c) and
# #                                         ("photo" in _cap_toks_c or "photograph" in _cap_toks_c
# #                                          or "drawing" in _cap_toks_c or "painting" in _cap_toks_c))

# #                             if _cap_colours and not _matching and not _is_mono:
# #                                 # Caption names a DIFFERENT colour → wrong subject
# #                                 logger.info(
# #                                     f"🎨 OD colour-filter: id={_img.id} "
# #                                     f"query_colours={_q_colours} cap_colours={_cap_colours} → skip"
# #                                 )
# #                                 continue
# #                             elif not _cap_colours:
# #                                 # Caption has no colour — defer to CLIP+pixel scoring
# #                                 # Don't add to OD results; let it fall through to CLIP
# #                                 logger.debug(
# #                                     f"🎨 OD colour-defer: id={_img.id} "
# #                                     f"no colour in caption → CLIP will score by pixel"
# #                                 )
# #                                 continue
# #                             elif _matching:
# #                                 # Confirmed colour match → boost score
# #                                 score = min(score + 5, 97.0)

# #                 _seen_od.add(_img.id)
# #                 _seen_fn.add(fn)
# #                 _od_results.append(_od_row(_img, score))

# #             # Sort by score desc
# #             _od_results.sort(key=lambda x: x["score"], reverse=True)

# #             # FIX Bug1: dedup by caption text — catches same-content images uploaded
# #             # twice (different UUID filenames, so filename-dedup misses them).
# #             _seen_cap_text = set()
# #             _od_deduped    = []
# #             for _r in _od_results:
# #                 _ct = (_r.get("caption_short") or "").strip().lower()
# #                 if _ct and _ct in _seen_cap_text:
# #                     logger.info(f"🔁 OD dedup: skipping duplicate caption '{_ct[:50]}'")
# #                     continue
# #                 if _ct:
# #                     _seen_cap_text.add(_ct)
# #                 _od_deduped.append(_r)
# #             _od_results = _od_deduped

# #             if _od_results:
# #                 logger.info(f"✅ OD-search '{query}' → {len(_od_results)} results via scene_label/tags")
# #                 return {
# #                     "status": "found",
# #                     "query": query,
# #                     "count": len(_od_results),
# #                     "results": _od_results[:top_k],
# #                 }
# #         finally:
# #             _db_od.close()
# #         # OD found nothing → try caption text search for the animal/object words
# #         _caption_results = []
# #         _seen_cap = set()
# #         # Search caption text for query words directly
# #         _search_terms = list(_q_lower_words) + [v for k,v in QUERY_TO_COCO.items() if k in _q_lower_words]
# #         _search_terms = list(set(_search_terms))
# #         _db_cap = SessionLocal()
# #         try:
# #             for _img in _live(_db_cap).all():
# #                 # Search caption, scene_label, AND user tags
# #                 _cap_lower = (_img.caption_short or "").lower()
# #                 _det_lower = (_img.caption_detailed or "").lower()
# #                 _scene_lower = (_img.scene_label or "").lower()
# #                 _tags_lower = " ".join(_user_tags(_img)).lower()
# #                 _haystack = " ".join(filter(None,[_cap_lower, _det_lower, _scene_lower, _tags_lower]))
# #                 # Check if any search term appears anywhere
# #                 if _haystack and any(t in _haystack for t in _search_terms) and _img.id not in _seen_cap:
# #                     _seen_cap.add(_img.id)
# #                     _caption_results.append({
# #                         "id": _img.id,
# #                         "filename": _img_url(_img.filename),
# #                         "score": 80.0,
# #                         "timestamp": _img.timestamp.isoformat() if _img.timestamp else None,
# #                         "caption_short": _img.caption_short or "",
# #                         "person_count": _img.person_count or 0,
# #                         "dominant_emotion": _img.dominant_emotion or "",
# #                         "quality_level": _img.quality_level or "",
# #                         "quality_score": _img.quality_score or 0,
# #                         "aesthetic_score": _img.aesthetic_score or 0,
# #                         "user_tags": _user_tags(_img),
# #                         "photo_note": getattr(_img, "photo_note", "") or "",
# #                         "is_favorite": bool(_img.is_favorite),
# #                     })
# #         finally:
# #             _db_cap.close()
# #         if _caption_results:
# #             logger.info(f"✅ Caption-search '{query}' → {len(_caption_results)} results")
# #             return {
# #                 "status": "found", "query": query,
# #                 "count": len(_caption_results),
# #                 "results": sorted(_caption_results, key=lambda x: x["score"], reverse=True)[:top_k]
# #             }
# #         logger.info(f"⚠️ OD+Caption search found nothing for {_search_terms}, falling back to CLIP")

# #     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
# #     if query_emb is None or search_engine.index is None:
# #         return {"status": "error", "message": "No images indexed. Upload some photos first."}
# #     query_lower  = query.lower()
# #     # q_tokens_pre already defined above (before should_search_people pre-compute)

# #     # For person-name queries, clear sig_words so CLIP scoring
# #     # doesn't give caption bonus to irrelevant images (e.g. pig-in-sink for "Sadie Sink")
# #     sig_words    = [] if should_search_people else [w for w in processed_query.lower().split() if len(w) > 2]
# #     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in query_lower]

# #     # ── Compound word expansion ────────────────────────────────────────────────
# #     # BLIP captions write "iron man" (two words) but users search "ironman" (one).
# #     # Expand known compounds so all keyword paths find the image.
# #     _COMPOUND_SPLITS = {
# #         "ironman":     ["iron","man"],      "spiderman":    ["spider","man"],
# #         "batman":      ["bat","man"],       "superman":     ["super","man"],
# #         "blackwidow":  ["black","widow"],   "antman":       ["ant","man"],
# #         "warmachine":  ["war","machine"],   "blackpanther": ["black","panther"],
# #         "ironheart":   ["iron","heart"],    "deadpool":     ["dead","pool"],
# #         "daredevil":   ["dare","devil"],    "spiderverse":  ["spider","verse"],
# #     }
# #     if sig_words:
# #         _exp = []
# #         for _sw in sig_words:
# #             _exp.append(_sw)
# #             if _sw in _COMPOUND_SPLITS:
# #                 _exp.extend(_COMPOUND_SPLITS[_sw])
# #         sig_words = list(dict.fromkeys(_exp))

# #     # Also expand the processed_query itself so CLIP gets "iron man" instead of "ironman"
# #     _pq_words = processed_query.lower().split()
# #     _pq_expanded = []
# #     for _pw in _pq_words:
# #         if _pw in _COMPOUND_SPLITS:
# #             _pq_expanded.extend(_COMPOUND_SPLITS[_pw])
# #         else:
# #             _pq_expanded.append(_pw)
# #     if _pq_expanded != _pq_words:
# #         processed_query = " ".join(_pq_expanded)
# #         logger.info(f"🔤 Compound expand: '{query}' → CLIP query='{processed_query}'")
# #     db = SessionLocal()
# #     try:
# #         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k)
# #         # ── Person / actor name search ─────────────────────────────────────
# #         # ── Person name search — only for name-like queries ────────────────
# #         # CRITICAL: Only run if query looks like a person name, NOT for
# #         # descriptive queries like "woman in black dress".
# #         # Rules:
# #         #   - Must have at least one word >= 4 chars that isn't a common word
# #         #   - Must not be a purely descriptive query (too many common words)
# #         #   - Person name matching uses WHOLE WORD matching only (not substring)

# #         DESCRIPTOR_WORDS = {
# #             # articles / prepositions / conjunctions
# #             "a","an","the","in","on","at","of","for","with","by","to","from",
# #             "into","onto","over","under","above","below","near","next","beside",
# #             "and","or","but","not","nor","yet","so","both","either","neither",
# #             # verbs
# #             "is","are","was","were","be","been","being","has","have","had",
# #             "do","does","did","will","would","could","should","may","might",
# #             "can","wearing","holding","standing","sitting","posing","looking",
# #             "running","walking","smiling","laughing","dancing","jumping","riding",
# #             # common adjectives
# #             "long","short","tall","small","big","large","little","young","old",
# #             "beautiful","pretty","lovely","nice","great","black","white","red",
# #             "blue","green","yellow","orange","purple","pink","brown","gray",
# #             "dark","light","bright","dark","shiny","fluffy","cute","funny",
# #             # people descriptors (not names)
# #             "woman","man","girl","boy","lady","guy","person","people","female",
# #             "male","human","child","adult","baby","teenager","player","actor",
# #             # scene/object words — CRITICAL: prevents these being treated as names
# #             "photo","image","picture","standing","next","another","front","back",
# #             "window","field","camera","dress","suit","shirt","jacket","hair",
# #             "wearing","sitting","holding","looking","posing","smiling",
# #             # common household/body nouns that appear in captions
# #             "sink","table","chair","floor","wall","door","room","house","home",
# #             "hand","face","head","hair","eyes","mouth","nose","body","back",
# #             "water","grass","tree","flower","rock","snow","sand","road","street",
# #             "book","phone","glass","bowl","plate","ball","box","bag","hat",
# #             "food","cake","pizza","coffee","wine","beer","milk","soup",
# #             "park","beach","mountain","forest","city","stage","roof","pool",
# #             "movie","film","show","song","music","sport","game","team","show",
# #             "scene","time","year","day","night","morning","evening","world",
# #         }

# #         raw_q      = query.lower().strip()
# #         q_tokens   = raw_q.split()
# #         # Words that could plausibly be part of a name (long, not descriptive)
# #         # For short queries (≤3 words like "tom holland"), allow 3-char words
# #         min_name_len = 3 if len(q_tokens) <= 3 else 4
# #         name_candidates = [
# #             w for w in q_tokens
# #             if len(w) >= min_name_len
# #             and w not in DESCRIPTOR_WORDS
# #             and w.isalpha()
# #         ]

# #         descriptor_count = sum(1 for w in q_tokens if w in DESCRIPTOR_WORDS)
# #         is_descriptive   = descriptor_count > len(q_tokens) * 0.5   # strict: >50% descriptors needed
# #         should_search_people = bool(name_candidates) and not is_descriptive

# #         vector_ids   = {r["id"] for r in results}
# #         people_rows  = db.query(Person).all() if should_search_people else []

# #         # ── 1. Match against named person clusters ────────────────────────
# #         matched_people = []
# #         if should_search_people:
# #             for p in people_rows:
# #                 pname_raw = (p.name or "").strip()
# #                 pn        = pname_raw.lower()
# #                 if not pn:
# #                     continue
# #                 # Skip default "Person N" labels
# #                 if re.match(r"^person\s+\d+$", pn):
# #                     continue
# #                 # Skip OCR garbage: all-caps multi-word, or contains digits,
# #                 # or contains non-alpha chars (PETRONAS INEQS, Hak IINDIA, etc.)
# #                 if re.search(r"\d", pname_raw):
# #                     continue   # contains digit → OCR garbage
# #                 p_words = pn.split()
# #                 # If all words are UPPERCASE in original → likely OCR label not a name
# #                 if all(w == w.upper() and len(w) > 1 for w in pname_raw.split()):
# #                     continue
# #                 # Whole-word matching ONLY — "in" must NOT match inside "india"
# #                 # Use word boundary: check if name word appears as standalone token
# #                 p_parts = [pt for pt in p_words if len(pt) >= 3]
# #                 query_tokens_set = set(q_tokens)
# #                 matched = False
# #                 for pt in p_parts:
# #                     if pt in query_tokens_set:        # exact token match
# #                         matched = True; break
# #                 for nc in name_candidates:
# #                     if nc in p_words:                 # query name-word in person name
# #                         matched = True; break
# #                     if pn == nc or nc == pn:
# #                         matched = True; break
# #                     # Allow partial first/last name: "vijay" matches "vijay kumar"
# #                     if any(nc == pw for pw in p_words):
# #                         matched = True; break
# #                 if matched:
# #                     matched_people.append(p)

# #             # ── 2. Caption / OCR / VQA text search (works WITHOUT renaming) ──
# #             # Only run if face-cluster lookup found no results yet
# #             # (avoids flooding with false positives from common words in OCR)
# #             COMMON_WORDS = {
# #                 "the","and","for","are","but","not","you","all","any","can",
# #                 "had","her","was","one","our","out","day","get","has","him",
# #                 "his","how","its","may","new","now","old","see","two","way",
# #                 "who","did","man","men","two","use","she","him","this","that",
# #                 "with","from","they","have","more","will","home","also","into",
# #                 "over","time","very","when","your","come","here","just","like",
# #                 "long","make","many","most","some","them","than","then","these",
# #                 "well","were","what","each","much","both","been","only","same",
# #                 "india","team","cup","match","player","game","year","last",
# #             }
# #             # Only do caption search for proper-noun-looking queries
# #             # (single words >= 4 chars that aren't common English words)
# #             search_words = [w for w in name_candidates if len(w) >= 4 and w not in COMMON_WORDS]
# #             if search_words and not matched_people:
# #                 for img in _live(db).all():
# #                     if img.id in vector_ids:
# #                         continue
# #                     vqa_person = ""
# #                     try:
# #                         vqa_d = json.loads(img.caption_vqa or "{}")
# #                         vqa_person = vqa_d.get("person", "") or vqa_d.get("subject", "")
# #                     except Exception:
# #                         pass
# #                     hay = " ".join(filter(None, [
# #                         img.caption_short or "",
# #                         img.caption_detailed or "",
# #                         img.ocr_text_enhanced or "",
# #                         img.scene_label or "",
# #                         vqa_person,
# #                         " ".join(_user_tags(img)),
# #                     ])).lower()
# #                     # Require ALL search words to match (avoids false positives)
# #                     if all(w in hay for w in search_words):
# #                         results.append({
# #                             "id": img.id, "filename": _img_url(img.filename),
# #                             "score": 65.0,
# #                             "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                             "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# #                             "person_count": img.person_count or 0,
# #                             "caption_short": img.caption_short or "",
# #                             "user_tags": _user_tags(img),
# #                             "photo_note": getattr(img, "photo_note", "") or "",
# #                         })
# #                         vector_ids.add(img.id)

# #             if matched_people:
# #                 # CRITICAL: When we found a person match, REPLACE CLIP results entirely.
# #                 # CLIP results for name queries are unreliable (e.g. "sadie sink" → pig-in-sink).
# #                 # Only return face-cluster matches + caption/OCR name matches.
# #                 person_results = []
# #                 person_ids_seen = set()

# #                 for person in matched_people:
# #                     logger.info(f"🧑 Searching for person: '{person.name}' (id={person.id})")
# #                     face_records = db.query(DBFace).filter(DBFace.person_id == person.id).all()
# #                     face_img_ids = list({f.image_id for f in face_records if f.image_id})
# #                     logger.info(f"  → {len(face_records)} face records, {len(face_img_ids)} unique image_ids")

# #                     # 1. Face-cluster matched images (highest confidence = 95)
# #                     if face_img_ids:
# #                         for img in _live(db).filter(DBImage.id.in_(face_img_ids)).all():
# #                             if img.id not in person_ids_seen:
# #                                 person_results.append({
# #                                     "id": img.id, "filename": _img_url(img.filename),
# #                                     "score": 95.0,
# #                                     "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                                     "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# #                                     "person_count": img.person_count or 0,
# #                                     "caption_short": img.caption_short or "",
# #                                     "user_tags": _user_tags(img),
# #                                     "photo_note": getattr(img, "photo_note", "") or "",
# #                                     "matched_person": person.name,
# #                                 })
# #                                 person_ids_seen.add(img.id)

# #                     # 2. Caption/OCR text search for the FULL NAME as a phrase only
# #                     # Use full name phrase — prevents "sink" matching pig captions
# #                     name_kw = person.name.lower()
# #                     name_parts = name_kw.split()  # ["sadie", "sink"]
# #                     for img in _live(db).all():
# #                         if img.id in person_ids_seen:
# #                             continue
# #                         hay = " ".join(filter(None, [
# #                             img.caption_short or "", img.caption_detailed or "",
# #                             img.caption_vqa or "", img.ocr_text_enhanced or "",
# #                             img.scene_label or ""
# #                         ])).lower()
# #                         # Must contain FULL NAME as phrase OR all parts together
# #                         # NOT individual words (prevents "sink" matching pig captions)
# #                         if name_kw in hay or (len(name_parts) > 1 and all(p in hay for p in name_parts)):
# #                             # Extra check: if name has common words (like "sink"),
# #                             # require the rare part to appear near human context
# #                             person_results.append({
# #                                 "id": img.id, "filename": _img_url(img.filename),
# #                                 "score": 75.0,
# #                                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
# #                                 "person_count": img.person_count or 0,
# #                                 "caption_short": img.caption_short or "",
# #                                 "user_tags": _user_tags(img),
# #                                 "photo_note": getattr(img, "photo_note", "") or "",
# #                                 "matched_person": person.name,
# #                             })
# #                             person_ids_seen.add(img.id)

# #                     if not face_img_ids:
# #                         logger.warning(f"  ⚠️ No face records for '{person.name}' — only caption search. Run Re-index AI.")

# #                 if person_results:
# #                     results = sorted(person_results, key=lambda x: x["score"], reverse=True)
# #                     return {"status": "found", "query": query, "count": len(results), "results": results[:top_k]}
# #                 else:
# #                     # Person found in DB but no photos linked yet — tell user to re-index
# #                     names = ", ".join(p.name for p in matched_people)
# #                     return {
# #                         "status": "not_found",
# #                         "message": f"Found person '{names}' but no photos linked. Click Re-index AI to rebuild face clusters.",
# #                         "people_suggestions": [{"id": p.id, "name": p.name, "cover": None} for p in matched_people],
# #                     }

# #         # ── No person found in DB for a name query → try content search first ──
# #         if should_search_people and not matched_people:
# #             # Before giving up, attempt 3 fallbacks in priority order:
# #             # 1. Tag fast-path results (user manually tagged this image)
# #             # 2. Caption/OCR keyword search (works for ironman, thor, spider, etc.)
# #             # 3. Only if both fail → show "not found" with people suggestions

# #             # ── Fallback 1: tags ──────────────────────────────────────────────
# #             if _tag_fast_results:
# #                 logger.info(f"🏷️ Person-path fallback → {len(_tag_fast_results)} tag results for '{query}'")
# #                 return {
# #                     "status": "found", "query": query,
# #                     "count": len(_tag_fast_results),
# #                     "results": _tag_fast_results[:top_k],
# #                 }

# #             # ── Fallback 2: caption + OCR keyword search ──────────────────────
# #             # Covers: "ironman", "thor", "spider", "kpop", "user" (OCR), etc.
# #             # Use the original query words (NOT processed_query which strips non-CLIP terms)
# #             _kw_query_words = [w for w in query.lower().split() if len(w) > 2]
# #             _kw_results_fb  = []
# #             _kw_seen_fb     = set()
# #             _kw_seen_cap_fb = set()
# #             if _kw_query_words:
# #                 for _img_fb in _live(db).all():
# #                     if _img_fb.id in _kw_seen_fb:
# #                         continue
# #                     # Build search haystack: caption + OCR + scene_label + tags
# #                     _fb_hay = " ".join(filter(None, [
# #                         (_img_fb.caption_short    or "").lower(),
# #                         (_img_fb.caption_detailed or "").lower(),
# #                         (_img_fb.ocr_text_enhanced or "").lower(),
# #                         (_img_fb.scene_label      or "").lower(),
# #                         " ".join(_user_tags(_img_fb)).lower(),
# #                     ]))
# #                     if not _fb_hay:
# #                         continue
# #                     # ALL query words must appear (strict — avoids false positives)
# #                     if all(w in _fb_hay for w in _kw_query_words):
# #                         _ct_fb = (_img_fb.caption_short or "").strip().lower()
# #                         if _ct_fb and _ct_fb in _kw_seen_cap_fb:
# #                             continue
# #                         if _ct_fb:
# #                             _kw_seen_cap_fb.add(_ct_fb)
# #                         _kw_seen_fb.add(_img_fb.id)
# #                         _kw_results_fb.append({
# #                             "id": _img_fb.id,
# #                             "filename": _img_url(_img_fb.filename) or "",
# #                             "score": 80.0,
# #                             "timestamp": _img_fb.timestamp.isoformat() if _img_fb.timestamp else None,
# #                             "caption_short": _img_fb.caption_short or "",
# #                             "person_count": _img_fb.person_count or 0,
# #                             "dominant_emotion": _img_fb.dominant_emotion or "",
# #                             "quality_level": _img_fb.quality_level or "",
# #                             "quality_score": _img_fb.quality_score or 0,
# #                             "aesthetic_score": _img_fb.aesthetic_score or 0,
# #                             "user_tags": _user_tags(_img_fb),
# #                             "photo_note": getattr(_img_fb, "photo_note", "") or "",
# #                             "is_favorite": bool(_img_fb.is_favorite),
# #                             "scene_label": _img_fb.scene_label or "",
# #                             "width": _img_fb.width, "height": _img_fb.height,
# #                         })
# #             if _kw_results_fb:
# #                 logger.info(f"📝 Person-path caption/OCR fallback → {len(_kw_results_fb)} results for '{query}'")
# #                 return {
# #                     "status": "found", "query": query,
# #                     "count": len(_kw_results_fb),
# #                     "results": sorted(_kw_results_fb, key=lambda x: x["score"], reverse=True)[:top_k],
# #                 }

# #             # ── Fallback 3: truly nothing found → people suggestions ──────────
# #             all_people = db.query(Person).all()
# #             suggestions = []
# #             for p in all_people:
# #                 if re.match(r"^person\s+\d+$", (p.name or "").lower()): continue
# #                 face_recs = db.query(DBFace).filter(DBFace.person_id == p.id).limit(1).all()
# #                 if face_recs and face_recs[0].image_id:
# #                     cover_img = db.query(DBImage).filter(DBImage.id == face_recs[0].image_id).first()
# #                     if cover_img:
# #                         suggestions.append({"id": p.id, "name": p.name, "cover": _img_url(cover_img.filename)})
# #             return {
# #                 "status": "not_found",
# #                 "message": f"No photos matched '{query}'.",
# #                 "people_suggestions": suggestions[:12],
# #             }

# #         # Fallback: discard any CLIP junk when person matched
# #         if matched_people:
# #             person_results = [r for r in results if r.get("matched_person")]
# #             if person_results:
# #                 results = person_results
# #             results.sort(key=lambda x: x["score"], reverse=True)

# #         # ── Skip keyword fallback when we already have person results ──────────
# #         # Person name searches should ONLY return person face images, not
# #         # generic CLIP results or keyword matches for ambiguous words like "sink"
# #         skip_kw_fallback = bool(matched_people)

# #         if sig_words and not skip_kw_fallback:
# #             vector_ids = {r["id"] for r in results}
# #             KW_STOP = {"the","a","an","in","on","at","of","for","with","by","to",
# #                        "from","is","are","was","were","be","and","or","but","not"}
# #             meaningful = [w for w in sig_words if w not in KW_STOP and len(w) > 2]

# #             # ── Rebuild contradiction context for filtering ────────────────
# #             COLOURS_KW = {"black","white","red","blue","green","yellow","orange",
# #                           "purple","pink","brown","gray","grey","golden","silver"}
# #             CLOTHING_KW = {"dress","suit","shirt","jacket","sari","skirt","gown",
# #                            "uniform","coat","blouse","saree","lehenga"}
# #             ANIMALS_KW = {"horse","cow","dog","cat","bird","fox","tiger","lion",
# #                           "bear","wolf","pig","rabbit","fish","duck","frog","otter",
# #                           "elephant","monkey","deer","sheep","goat","snake","chicken",
# #                           "kitten","puppy","calf","hamster","squirrel","raccoon"}
# #             FEMALE_KW = {"woman","girl","lady","female"}
# #             MALE_KW   = {"man","boy","guy","male"}
# #             q_set_kw      = set(meaningful)
# #             q_colours_kw  = COLOURS_KW  & q_set_kw
# #             q_clothing_kw = CLOTHING_KW & q_set_kw
# #             q_animals_kw  = ANIMALS_KW  & q_set_kw
# #             q_gender_kw   = (FEMALE_KW | MALE_KW) & q_set_kw

# #             def _kw_ok(caption_text):
# #                 """Return True if caption doesn't contradict the query."""
# #                 cap = set(caption_text.split())
# #                 # Stemmed cap words
# #                 def s(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
# #                 cap_stem = {s(w) for w in cap} | cap

# #                 if q_animals_kw:
# #                     cap_animals = ANIMALS_KW & cap_stem
# #                     if cap_animals and not (cap_animals & q_animals_kw):
# #                         return False   # different animal
# #                     # Also: if caption has PEOPLE words but NO animals at all → exclude
# #                     # (e.g. "man standing in a field" shouldn't appear for "dog" query)
# #                     PEOPLE_KW = {"man","woman","person","girl","boy","people","guy",
# #                                  "lady","male","female","player","actor","celebrity"}
# #                     if not cap_animals and (PEOPLE_KW & cap_stem):
# #                         return False   # people-only caption for animal query
# #                 if q_colours_kw:
# #                     cap_col = COLOURS_KW & cap
# #                     contradicting = cap_col - q_colours_kw
# #                     matching      = cap_col & q_colours_kw
# #                     if contradicting and not matching:
# #                         return False   # only wrong colours
# #                 if q_clothing_kw:
# #                     cap_cl = CLOTHING_KW & cap_stem
# #                     if cap_cl and not (cap_cl & q_clothing_kw):
# #                         return False   # different clothing
# #                 if q_gender_kw:
# #                     q_f = bool(FEMALE_KW & q_gender_kw)
# #                     q_m = bool(MALE_KW   & q_gender_kw)
# #                     if q_f and (MALE_KW & cap) and not (FEMALE_KW & cap):
# #                         return False
# #                     if q_m and (FEMALE_KW & cap) and not (MALE_KW & cap):
# #                         return False
# #                 return True

# #             kw_candidates = _live(db).filter(
# #                 DBImage.caption_short.isnot(None) | DBImage.caption_detailed.isnot(None)
# #             ).all()
# #             for img in kw_candidates:
# #                 if img.id in vector_ids:
# #                     continue
# #                 caption_text = " ".join(filter(None,[
# #                     img.caption_short or "", img.caption_detailed or ""
# #                 ])).lower()
# #                 haystack = caption_text + " " + " ".join(filter(None,[
# #                     img.ocr_text_enhanced or "", img.scene_label or "",
# #                     " ".join(_user_tags(img))
# #                 ])).lower()
# #                 if not meaningful:
# #                     continue
# #                 # Apply contradiction filter FIRST
# #                 if not _kw_ok(caption_text):
# #                     continue
# #                 # Compound word matching: "ironman" counts as matched if both
# #                 # "iron" AND "man" are present (even as separate words)
# #                 def _word_in_haystack(w, hs):
# #                     if w in hs:
# #                         return True
# #                     parts = _COMPOUND_SPLITS.get(w)
# #                     if parts and all(p in hs for p in parts):
# #                         return True
# #                     return False
# #                 # For 1-2 meaningful words: ALL must match
# #                 # For 3+: majority (≥70%) must match
# #                 matched = sum(1 for w in meaningful if _word_in_haystack(w, haystack))
# #                 threshold = len(meaningful) if len(meaningful) <= 2 else max(2, len(meaningful) * 0.70)
# #                 if matched >= threshold:
# #                     results.append({
# #                         "id": img.id, "filename": _img_url(img.filename), "score": 32.0,
# #                         "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                         "person_count": img.person_count or 0,
# #                         "caption_short": img.caption_short or "",
# #                         "caption_detailed": img.caption_detailed or "",
# #                         "quality_level": img.quality_level or "",
# #                         "quality_score": img.quality_score or 0,
# #                         "dominant_emotion": img.dominant_emotion or "",
# #                         "face_emotion_count": img.face_emotion_count or 0,
# #                         "aesthetic_score": img.aesthetic_score or 0,
# #                         "aesthetic_rating": img.aesthetic_rating or "",
# #                         "is_favorite": bool(img.is_favorite),
# #                         "user_tags": _user_tags(img),
# #                         "photo_note": getattr(img, "photo_note", "") or "",
# #                         "scene_label": img.scene_label or "",
# #                         "width": img.width, "height": img.height,
# #                     })
# #             results.sort(key=lambda x: x["score"], reverse=True)
# #             results = results[:top_k]

# #         elif sig_words and skip_kw_fallback:
# #             # Person search mode: drop any CLIP results that aren't person matches
# #             # (e.g. "sadie sink" → remove literal sink images from CLIP)
# #             person_img_ids = {r["id"] for r in results if r.get("matched_person")}
# #             if person_img_ids:
# #                 results = [r for r in results if r.get("matched_person") or r["id"] in person_img_ids]
# #             results.sort(key=lambda x: x["score"], reverse=True)
# #             results = results[:top_k]
# #         if not results:
# #             # If CLIP found nothing but the tag fast-path did, return tag results directly
# #             if _tag_fast_results:
# #                 logger.info(f"🏷️ CLIP found nothing — returning {len(_tag_fast_results)} tag-matched results")
# #                 return {
# #                     "status": "found", "query": query,
# #                     "count": len(_tag_fast_results),
# #                     "results": _tag_fast_results[:top_k],
# #                 }
# #             # Return people suggestions so user can identify who to search
# #             people_suggestions = []
# #             if name_candidates:
# #                 all_people = db.query(Person).all()
# #                 for p in all_people:
# #                     face_recs = db.query(DBFace).filter(DBFace.person_id == p.id).limit(1).all()
# #                     if face_recs and face_recs[0].image_id:
# #                         cover_img = db.query(DBImage).filter(DBImage.id == face_recs[0].image_id).first()
# #                         if cover_img:
# #                             people_suggestions.append({
# #                                 "id": p.id,
# #                                 "name": p.name,
# #                                 "cover": _img_url(cover_img.filename)
# #                             })
# #             return {
# #                 "status": "not_found",
# #                 "message": f"No images matched '{query}'.",
# #                 "people_suggestions": people_suggestions[:12],
# #             }
# #         # ── Merge tag fast-path results (score=97) before dedup ─────────────
# #         # Tag-matched images that weren't already in CLIP results get prepended
# #         # at the top so they always appear first regardless of CLIP score.
# #         if _tag_fast_results:
# #             existing_ids = {r["id"] for r in results}
# #             for _tr in _tag_fast_results:
# #                 if _tr["id"] not in existing_ids:
# #                     results.insert(0, _tr)
# #                     existing_ids.add(_tr["id"])
# #                 else:
# #                     # Already in results — boost its score to 97
# #                     for _r in results:
# #                         if _r["id"] == _tr["id"]:
# #                             _r["score"] = max(_r.get("score", 0), 97.0)
# #                             break
# #             results.sort(key=lambda x: x["score"], reverse=True)

# #         # ── BUGFIX: Final dedup by filename ─────────────────────────────────
# #         # Catches re-indexed duplicate DB rows (different id, same physical file)
# #         # that bypass the earlier id-based dedup in _score_candidates.
# #         _seen_fn_final  = set()
# #         _seen_cap_final = set()   # also dedup by caption — catches same-content re-uploads
# #         _deduped_final  = []
# #         for _r in results:
# #             _fn  = _r.get("filename") or ""
# #             _ct  = (_r.get("caption_short") or "").strip().lower()
# #             if _fn and _fn in _seen_fn_final:
# #                 continue
# #             # Caption dedup: skip if identical caption already in results
# #             # (keep first = highest scored; favourites get boosted by tag merge so appear first)
# #             if _ct and _ct in _seen_cap_final:
# #                 continue
# #             if _fn:
# #                 _seen_fn_final.add(_fn)
# #             if _ct:
# #                 _seen_cap_final.add(_ct)
# #             _deduped_final.append(_r)
# #         results = _deduped_final

# #         return {"status": "found", "query": query, "count": len(results), "results": results}
# #     finally:
# #         db.close()

# # @app.post("/search/describe")
# # def search_by_description(description: str = Form(...), top_k: int = Form(20)):
# #     if not description or not description.strip():
# #         return {"status": "error", "message": "Description empty"}
# #     if search_engine.index is None:
# #         return {"status": "error", "message": "No images indexed."}
# #     desc = _clean_query(description.strip())
# #     prompts = [desc, f"a photo of {desc}", f"an image showing {desc}", f"a picture of {desc}", f"{desc} photograph"]
# #     embs = [e for p in prompts for e in [search_engine.get_text_embedding(p, use_prompt_ensemble=False)] if e is not None]
# #     if not embs:
# #         return {"status": "error", "message": "Could not encode description."}
# #     query_emb = np.mean(embs, axis=0).astype("float32")
# #     norm = np.linalg.norm(query_emb)
# #     if norm > 1e-8:
# #         query_emb /= norm
# #     sig_words    = [w for w in desc.lower().split() if len(w) > 2]
# #     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in desc.lower()]
# #     db = SessionLocal()
# #     try:
# #         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k)
# #         if not results:
# #             return {"status": "not_found", "message": f"No images matched description '{desc}'."}
# #         return {"status": "found", "query": desc, "count": len(results), "results": results}
# #     finally:
# #         db.close()

# # @app.post("/search/hybrid")
# # async def search_hybrid(query: str = Form(""), file: UploadFile = File(None), text_weight: float = Form(0.6), image_weight: float = Form(0.4), top_k: int = Form(20)):
# #     if search_engine.index is None:
# #         return {"status": "error", "message": "No images indexed."}
# #     has_text  = query.strip() != ""
# #     has_image = file is not None and file.filename
# #     if not has_text and not has_image:
# #         return {"status": "error", "message": "Provide at least a query or an image."}
# #     if has_text and has_image:
# #         total_w = text_weight + image_weight
# #         if total_w <= 0:
# #             text_weight, image_weight = 0.6, 0.4
# #             total_w = 1.0
# #         text_weight /= total_w; image_weight /= total_w
# #     elif has_text:
# #         text_weight, image_weight = 1.0, 0.0
# #     else:
# #         text_weight, image_weight = 0.0, 1.0
# #     text_emb = None
# #     if has_text:
# #         text_emb = search_engine.get_text_embedding(_clean_query(query), use_prompt_ensemble=True)
# #     img_emb = None
# #     if has_image:
# #         import tempfile
# #         ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
# #         with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
# #             tmp_path = tmp.name
# #             shutil.copyfileobj(file.file, tmp)
# #         try:
# #             img_emb = search_engine.get_image_embedding(tmp_path)
# #         finally:
# #             try: os.remove(tmp_path)
# #             except Exception: pass
# #     if text_emb is not None and img_emb is not None:
# #         query_emb, extra_emb = text_emb, img_emb
# #     elif text_emb is not None:
# #         query_emb, extra_emb = text_emb, None
# #     elif img_emb is not None:
# #         query_emb, extra_emb = img_emb, None; text_weight, image_weight = 1.0, 0.0
# #     else:
# #         return {"status": "error", "message": "Could not encode query."}
# #     sig_words    = [w for w in query.lower().split() if len(w) > 2]
# #     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in query.lower()]
# #     db = SessionLocal()
# #     try:
# #         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k, extra_emb=extra_emb, text_weight=text_weight, image_weight=image_weight)
# #         if not results:
# #             return {"status": "not_found", "message": "No images matched."}
# #         return {"status": "found", "query": query, "count": len(results), "text_weight": text_weight, "image_weight": image_weight, "results": results}
# #     finally:
# #         db.close()

# # @app.post("/search/voice")
# # def voice_search_legacy(duration: int = Form(5)):
# #     try:
# #         transcribed = voice_engine.listen_and_transcribe(duration=duration)
# #         if not transcribed or not transcribed.strip():
# #             return {"status": "error", "message": "Could not hear anything."}
# #         result = search(query=transcribed.strip(), top_k=20)
# #         result["transcribed"] = transcribed.strip()
# #         return result
# #     except Exception as e:
# #         return {"status": "error", "message": f"Voice search failed: {str(e)}"}

# # @app.post("/search/image")
# # async def search_by_image(file: UploadFile = File(...), top_k: int = Form(20)):
# #     if search_engine.index is None:
# #         return {"status": "error", "message": "No images indexed."}
# #     ext = os.path.splitext(file.filename or "")[1].lower()
# #     if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
# #         raise HTTPException(status_code=400, detail="Unsupported format.")
# #     import tempfile
# #     with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
# #         tmp_path = tmp.name
# #         shutil.copyfileobj(file.file, tmp)
# #     try:
# #         query_emb = search_engine.get_image_embedding(tmp_path)
# #     finally:
# #         try: os.remove(tmp_path)
# #         except Exception: pass
# #     if query_emb is None:
# #         return {"status": "error", "message": "Could not process image."}
# #     total = search_engine.index.ntotal
# #     q = query_emb.reshape(1, -1).astype("float32")
# #     faiss.normalize_L2(q)
# #     distances, indices = search_engine.index.search(q, min(top_k * 3, total))
# #     db = SessionLocal()
# #     try:
# #         results = []
# #         for dist, idx in zip(distances[0], indices[0]):
# #             if idx == -1: continue
# #             score = float(dist)
# #             if score < 0.45: break
# #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# #             if not img or img.is_trashed: continue
# #             results.append({"id": img.id, "filename": _img_url(img.filename), "score": round(score * 100, 2), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None, "person_count": img.person_count or 0})
# #         results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
# #         if not results:
# #             return {"status": "not_found", "message": "No visually similar images found."}
# #         return {"status": "found", "count": len(results), "results": results}
# #     finally:
# #         db.close()

# # @app.post("/search/color")
# # def search_by_color(color: str = Form(...), top_k: int = Form(20)):
# #     """
# #     Color search using HSV hue matching instead of raw RGB distance.
# #     Much more accurate — a blue sky photo scores high for 'blue' regardless
# #     of how many green trees are also in the frame.
# #     """
# #     import colorsys

# #     # ── Hue ranges in degrees (0-360) for each color ────────────────────────
# #     # Each entry: (hue_center, hue_half_width, min_saturation, min_value)
# #     COLOR_HSV = {
# #         "red":    [(  0, 18, 0.35, 0.20), (360, 18, 0.35, 0.20)],  # red wraps around 0/360
# #         "orange": [( 25, 15, 0.45, 0.25)],
# #         "yellow": [( 55, 18, 0.40, 0.30)],
# #         "green":  [(120, 40, 0.30, 0.15)],
# #         "blue":   [(220, 45, 0.30, 0.15)],
# #         "purple": [(280, 30, 0.25, 0.15)],
# #         "pink":   [(330, 25, 0.25, 0.30)],
# #         "white":  None,   # special: high value, low saturation
# #         "black":  None,   # special: low value
# #         "gray":   None,   # special: low saturation
# #         "grey":   None,
# #         "brown":  [( 25, 18, 0.35, 0.15)],  # like orange but lower value
# #     }

# #     color_key = color.strip().lower()
# #     if color_key not in COLOR_HSV:
# #         color_key = next((k for k in COLOR_HSV if k in color_key), None)
# #     if not color_key:
# #         return {"status": "error", "message": f"Unknown color '{color}'."}

# #     def _color_score(r, g, b, ckey):
# #         """Score 0-100 how well an RGB pixel matches the target color."""
# #         r_, g_, b_ = (r or 0)/255.0, (g or 0)/255.0, (b or 0)/255.0
# #         h, s, v = colorsys.rgb_to_hsv(r_, g_, b_)
# #         hue_deg = h * 360.0

# #         if ckey in ("white", "grey", "gray"):
# #             # High brightness, low saturation
# #             if ckey == "white":
# #                 return max(0.0, (v - 0.75) / 0.25 * 100) * max(0.0, (0.25 - s) / 0.25)
# #             else:  # gray
# #                 sat_score = max(0.0, (0.20 - s) / 0.20)
# #                 val_score = max(0.0, 1.0 - abs(v - 0.50) / 0.50)
# #                 return sat_score * val_score * 100
# #         if ckey == "black":
# #             return max(0.0, (0.25 - v) / 0.25 * 100)

# #         ranges = COLOR_HSV[ckey]
# #         if not ranges:
# #             return 0.0

# #         best = 0.0
# #         for entry in ranges:
# #             hc, hw, min_s, min_v = entry
# #             # Hue distance (circular)
# #             diff = abs(hue_deg - hc)
# #             if diff > 180: diff = 360 - diff
# #             if diff > hw * 2:
# #                 continue
# #             hue_score  = max(0.0, 1.0 - diff / (hw * 2))
# #             sat_score  = min(1.0, max(0.0, (s - min_s) / (1.0 - min_s + 0.01)))
# #             val_score  = min(1.0, max(0.0, (v - min_v) / (1.0 - min_v + 0.01)))
# #             # brown special case: penalise high-value (bright) oranges
# #             if ckey == "brown":
# #                 val_score *= max(0.0, 1.0 - max(0.0, v - 0.55) / 0.45)
# #             score = hue_score * sat_score * val_score
# #             best = max(best, score)
# #         return round(best * 100, 2)

# #     db = SessionLocal()
# #     try:
# #         images = _live(db).filter(
# #             DBImage.avg_r != None, DBImage.avg_g != None, DBImage.avg_b != None
# #         ).all()
# #         if not images:
# #             return {"status": "not_found", "message": "No color data. Upload photos first."}

# #         scored = []
# #         for img in images:
# #             s = _color_score(img.avg_r, img.avg_g, img.avg_b, color_key)
# #             if s >= 10.0:  # low threshold — HSV matching is already precise
# #                 scored.append((s, img))

# #         scored.sort(key=lambda x: x[0], reverse=True)
# #         scored = scored[:top_k]

# #         if not scored:
# #             return {"status": "not_found", "message": f"No images matched color '{color}'."}
# #         return {
# #             "status": "found", "query": color, "count": len(scored),
# #             "results": [{
# #                 "id": img.id, "filename": _img_url(img.filename),
# #                 "score": round(s, 2),
# #                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                 "caption_short": img.caption_short or "",
# #                 "user_tags": _user_tags(img),
# #                 "dominant_emotion": img.dominant_emotion,
# #                 "person_count": img.person_count or 0,
# #             } for s, img in scored]
# #         }
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # REPROCESS EMOTIONS (fixes existing photos that have "neutral" default)
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.post("/reprocess-colors")
# # def reprocess_colors():
# #     """Recompute dominant color for all existing photos using weighted histogram."""
# #     import colorsys as _cs2
# #     from PIL import Image as PILImage2
# #     db = SessionLocal()
# #     try:
# #         images = _live(db).all()
# #         done = 0
# #         for img in images:
# #             fpath = img.original_path or os.path.join(IMAGE_DIR, img.filename)
# #             if not os.path.exists(fpath):
# #                 continue
# #             try:
# #                 pil = PILImage2.open(fpath).convert("RGB").resize((64,64))
# #                 arr2 = np.array(pil, dtype=np.float32)
# #                 r_v = arr2[:,:,0].flatten(); g_v = arr2[:,:,1].flatten(); b_v = arr2[:,:,2].flatten()
# #                 wts = np.array([
# #                     _cs2.rgb_to_hsv(r/255,g/255,b/255)[1] * _cs2.rgb_to_hsv(r/255,g/255,b/255)[2] + 0.01
# #                     for r,g,b in zip(r_v,g_v,b_v)
# #                 ])
# #                 tw = wts.sum()
# #                 img.avg_r = float(np.dot(r_v, wts)/tw)
# #                 img.avg_g = float(np.dot(g_v, wts)/tw)
# #                 img.avg_b = float(np.dot(b_v, wts)/tw)
# #                 done += 1
# #             except Exception as e:
# #                 logger.warning(f"Color reprocess failed {img.id}: {e}")
# #         db.commit()
# #         return {"status": "done", "updated": done}
# #     except Exception as e:
# #         db.rollback(); raise HTTPException(500, str(e))
# #     finally:
# #         db.close()


# # @app.post("/reprocess-emotions")
# # async def reprocess_emotions(background_tasks: BackgroundTasks, limit: int = 0):
# #     """
# #     Re-detect emotions on all uploaded images that still have the default
# #     'neutral' label or no emotion set. Runs in background so the API returns
# #     immediately.  limit=0 means process ALL.
# #     """
# #     db = SessionLocal()
# #     try:
# #         q = _live(db)
# #         if limit > 0:
# #             q = q.limit(limit)
# #         image_ids = [
# #             (img.id, img.original_path or os.path.join(IMAGE_DIR, img.filename))
# #             for img in q.all()
# #         ]
# #     finally:
# #         db.close()

# #     def _run_reprocess(pairs):
# #         done = 0
# #         for img_id, fpath in pairs:
# #             if not os.path.exists(fpath):
# #                 continue
# #             try:
# #                 # ── GATE: use InsightFace face count, not Faster R-CNN person_count ──
# #                 # person_count = 0 for close-up portraits (no full body visible).
# #                 # InsightFace accurately detects human faces regardless of body visibility.
# #                 # Cat faces, car headlights, turtles → InsightFace gives 0 → skip model.
# #                 _db_chk = SessionLocal()
# #                 try:
# #                     _face_count = _db_chk.query(DBFace).filter(DBFace.image_id == img_id).count()
# #                 except Exception:
# #                     _face_count = -1  # unknown → let it through
# #                 finally:
# #                     _db_chk.close()

# #                 if _face_count == 0:
# #                     # No real human faces → clear any previously wrong emotion label
# #                     _db_n = SessionLocal()
# #                     try:
# #                         _db_n.query(DBImage).filter(DBImage.id == img_id).update({
# #                             "dominant_emotion": "neutral", "face_emotion_count": 0,
# #                             "emotion_data": "[]",
# #                         }, synchronize_session=False)
# #                         _db_n.commit()
# #                     finally:
# #                         _db_n.close()
# #                     continue

# #                 ed = emotion_detection.detect_emotions(fpath)
# #                 # Confidence filter — same as _enrich_image
# #                 ed = [e for e in ed if e.get("confidence", 0) >= 0.50]
# #                 dominant = ed[0]["emotion"] if ed else "neutral"
# #                 db2 = SessionLocal()
# #                 try:
# #                     db2.query(DBImage).filter(DBImage.id == img_id).update({
# #                         "dominant_emotion":   dominant,
# #                         "face_emotion_count": len(ed),
# #                         "emotion_data":       _json.dumps(ed),
# #                     }, synchronize_session=False)
# #                     db2.commit()
# #                     done += 1
# #                 finally:
# #                     db2.close()
# #             except Exception as e:
# #                 logger.warning(f"Emotion reprocess failed for {img_id}: {e}")
# #         logger.info(f"✅ Emotion reprocess complete: {done}/{len(pairs)} images updated")

# #     background_tasks.add_task(_run_reprocess, image_ids)
# #     return {
# #         "status": "started",
# #         "message": f"Reprocessing emotions for {len(image_ids)} images in background",
# #         "count": len(image_ids)
# #     }


# # @app.post("/reprocess-names")
# # def reprocess_names():
# #     """
# #     Re-run auto-naming on ALL existing person clusters.
# #     Resets garbage names (ALL-CAPS brands, OCR fragments, short initials)
# #     back to "Person N", then re-runs the improved auto-naming logic.
# #     """
# #     db = SessionLocal()
# #     try:
# #         # ── Step 1: Reset bad auto-assigned names ───────────────────────────
# #         people_all = db.query(Person).all()
# #         reset_count = 0
# #         for i, p in enumerate(people_all):
# #             pn = (p.name or "").strip()
# #             # Never touch manually typed names (user renamed them — they know best)
# #             # We can't tell manual from auto, so use heuristics to detect garbage:
# #             should_reset = False

# #             if not pn or len(pn) < 2:
# #                 should_reset = True
# #             elif not any(c.isalpha() for c in pn):
# #                 should_reset = True
# #             else:
# #                 name_words = pn.split()
# #                 # Check every word in the name
# #                 bad_word_count = 0
# #                 for nw in name_words:
# #                     nw_clean = nw.rstrip(".,!?;:")
# #                     # ALL-CAPS word (brand name / jersey text)
# #                     if nw_clean.isalpha() and nw_clean == nw_clean.upper() and len(nw_clean) > 2:
# #                         bad_word_count += 1
# #                     # Too short (< 3 chars = initials / OCR fragment)
# #                     elif len(nw_clean) < 3:
# #                         bad_word_count += 1
# #                     # Known garbage word
# #                     elif nw_clean.lower() in _BLIP_NON_NAMES:
# #                         bad_word_count += 1
# #                     # No vowels (pure consonant = OCR garbage)
# #                     elif not any(c in "aeiouAEIOU" for c in nw_clean):
# #                         bad_word_count += 1
# #                 # If ANY word in the name is garbage → reset the whole name
# #                 if bad_word_count > 0:
# #                     should_reset = True

# #             if should_reset:
# #                 p.name = f"Person {i + 1}"
# #                 reset_count += 1

# #         db.commit()
# #         logger.info(f"🔄 Reset {reset_count} bad auto-names back to defaults")

# #         # ── Step 2: Re-run auto-naming with improved logic ──────────────────
# #         people_all = db.query(Person).all()
# #         person_map = {i: p.id for i, p in enumerate(people_all)}
# #         _auto_name_people(db, person_map)
# #         db.commit()

# #         named = db.query(Person).filter(~Person.name.startswith("Person ")).count()
# #         total = db.query(Person).count()
# #         return {
# #             "status": "done",
# #             "reset": reset_count,
# #             "named": named,
# #             "total": total,
# #             "message": f"{named}/{total} people named · {reset_count} bad names reset"
# #         }
# #     except Exception as e:
# #         db.rollback()
# #         raise HTTPException(500, str(e))
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # FACES / PEOPLE
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.get("/faces")
# # def get_faces(person_id: int = Query(None)):
# #     db = SessionLocal()
# #     try:
# #         if person_id:
# #             person = db.query(Person).filter(Person.id == person_id).first()
# #             if not person:
# #                 raise HTTPException(status_code=404)
# #             rows = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == person_id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
# #             cover  = _img_url(rows[0].filename) if rows else None
# #             images = [{"id": img.id, "filename": _img_url(img.filename), "thumbnail": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None} for img in rows]
# #             return {"id": person.id, "name": person.name, "face_count": len(images), "cover": cover, "images": images, "results": images}
# #         else:
# #             people  = db.query(Person).all()
# #             results = []
# #             for p in people:
# #                 imgs = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == p.id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
# #                 if not imgs: continue
# #                 results.append({"id": p.id, "name": p.name, "count": len(imgs), "cover": _img_url(imgs[0].filename)})
# #             return {"results": results, "count": len(results)}
# #     finally:
# #         db.close()

# # @app.get("/people/search")
# # def search_people_by_name(q: str = Query(...)):
# #     """Search people by name — returns matching persons with their photos."""
# #     db = SessionLocal()
# #     try:
# #         q_lower = q.lower().strip()
# #         if not q_lower:
# #             return {"results": []}
# #         all_people = db.query(Person).all()
# #         matched = []
# #         for p in all_people:
# #             pn = (p.name or "").lower().strip()
# #             if not pn or pn in ("unknown",):
# #                 continue
# #             parts = pn.split()
# #             if q_lower in pn or any(q_lower in part for part in parts) or any(part in q_lower for part in parts if len(part)>=2):
# #                 face_records = db.query(DBFace).filter(DBFace.person_id == p.id).all()
# #                 img_ids = list({f.image_id for f in face_records if f.image_id})
# #                 imgs = _live(db).filter(DBImage.id.in_(img_ids)).order_by(DBImage.timestamp.desc()).all() if img_ids else []
# #                 cover = _img_url(imgs[0].filename) if imgs else None
# #                 matched.append({
# #                     "id": p.id, "name": p.name, "count": len(imgs), "cover": cover,
# #                     "results": [{"id": img.id, "filename": _img_url(img.filename),
# #                                  "score": 92.0, "matched_person": p.name,
# #                                  "caption_short": img.caption_short or "",
# #                                  "timestamp": img.timestamp.isoformat() if img.timestamp else None,
# #                                  "user_tags": _user_tags(img),
# #                                  "person_count": img.person_count or 0} for img in imgs]
# #                 })
# #         return {"query": q, "count": len(matched), "results": matched}
# #     finally:
# #         db.close()


# # @app.get("/people/{person_id}")
# # def get_person(person_id: int):
# #     db = SessionLocal()
# #     try:
# #         person = db.query(Person).filter(Person.id == person_id).first()
# #         if not person:
# #             raise HTTPException(status_code=404)
# #         rows = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == person_id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
# #         cover  = _img_url(rows[0].filename) if rows else None
# #         images = [{"id": img.id, "filename": _img_url(img.filename), "thumbnail": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None} for img in rows]
# #         return {"id": person.id, "name": person.name, "face_count": len(images), "cover": cover, "images": images, "results": images}
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
# #     return {"status": "no_match"}


# # # ────────────────────────────────────────────────────────────────────────────
# # # ALBUMS
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.get("/albums/{album_id}")
# # def get_album_by_id(album_id: int):
# #     db = SessionLocal()
# #     try:
# #         album = db.query(Album).filter(Album.id == album_id).first()
# #         if not album:
# #             raise HTTPException(status_code=404, detail=f"Album {album_id} not found")
# #         images = _live(db).filter(DBImage.album_id == album_id).order_by(DBImage.timestamp).all()
# #         cover  = _img_url(images[0].filename) if images else None
# #         date_str = ""
# #         if album.start_date:
# #             date_str = album.start_date.strftime("%b %Y")
# #             if album.end_date and album.end_date.month != album.start_date.month:
# #                 date_str += f" – {album.end_date.strftime('%b %Y')}"
# #         img_list = [{"id": img.id, "filename": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None, "caption_short": img.caption_short, "ocr_text_enhanced": img.ocr_text_enhanced, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "aesthetic_score": img.aesthetic_score} for img in images]
# #         return {"id": album.id, "title": album.title, "type": album.type, "description": album.description, "date": date_str, "cover": cover, "start_date": album.start_date.isoformat() if album.start_date else None, "end_date": album.end_date.isoformat() if album.end_date else None, "image_count": len(images), "images": img_list, "results": img_list, "thumbnails": [_img_url(img.filename) for img in images[:4]]}
# #     finally:
# #         db.close()

# # @app.get("/albums")
# # def get_albums(album_id: int = Query(None)):
# #     db = SessionLocal()
# #     try:
# #         if album_id:
# #             return get_album_by_id(album_id)
# #         albums  = db.query(Album).all()
# #         results = []
# #         for a in albums:
# #             album_images = _live(db).filter(DBImage.album_id == a.id).all()
# #             # Show empty manual albums — user just created them
# #             if not album_images and a.type == "event":
# #                 continue  # hide empty auto-event albums, show empty manual ones
# #             date_str = ""
# #             if a.start_date:
# #                 date_str = a.start_date.strftime("%b %Y")
# #                 if a.end_date and a.end_date.month != a.start_date.month:
# #                     date_str += f" – {a.end_date.strftime('%b %Y')}"
# #             cover = _img_url(album_images[0].filename) if album_images else None
# #             results.append({"id": a.id, "title": a.title, "type": a.type or "manual",
# #                             "description": a.description, "date": date_str,
# #                             "cover": cover, "count": len(album_images),
# #                             "thumbnails": [_img_url(img.filename) for img in album_images[:4]]})
# #         return {"results": results, "count": len(results)}
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # ALBUM CRUD
# # # ────────────────────────────────────────────────────────────────────────────

# # @app.post("/albums/create")
# # def create_album(
# #     title: str = Form(...),
# #     description: str = Form(""),
# #     image_ids: str = Form(""),   # comma-separated image IDs to add immediately
# # ):
# #     """Create a new manual album, optionally with a set of images."""
# #     db = SessionLocal()
# #     try:
# #         album = Album(
# #             title=title.strip(),
# #             description=description.strip() or None,
# #             type="manual",
# #         )
# #         db.add(album); db.flush()

# #         added = 0
# #         if image_ids.strip():
# #             ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
# #             for img in _live(db).filter(DBImage.id.in_(ids)).all():
# #                 img.album_id = album.id
# #                 added += 1
# #             # Set album dates from images
# #             imgs = _live(db).filter(DBImage.album_id == album.id).all()
# #             timestamps = [i.timestamp for i in imgs if i.timestamp]
# #             if timestamps:
# #                 album.start_date = min(timestamps)
# #                 album.end_date   = max(timestamps)

# #         db.commit()
# #         logger.info(f"✅ Album created: '{title}' (id={album.id}, {added} images)")
# #         return {"status": "created", "id": album.id, "title": album.title, "image_count": added}
# #     except Exception as e:
# #         db.rollback(); raise HTTPException(500, str(e))
# #     finally:
# #         db.close()


# # @app.delete("/albums/{album_id}/delete")
# # def delete_album(album_id: int):
# #     """Delete an album (photos are kept, just unlinked from album)."""
# #     db = SessionLocal()
# #     try:
# #         album = db.query(Album).filter(Album.id == album_id).first()
# #         if not album:
# #             raise HTTPException(404, "Album not found")
# #         db.query(DBImage).filter(DBImage.album_id == album_id).update({"album_id": None})
# #         db.delete(album); db.commit()
# #         return {"status": "deleted", "id": album_id}
# #     except HTTPException: raise
# #     except Exception as e:
# #         db.rollback(); raise HTTPException(500, str(e))
# #     finally:
# #         db.close()


# # @app.delete("/albums/empty/cleanup")
# # def cleanup_empty_albums():
# #     """Delete all empty manual albums (useful after accidental duplicates)."""
# #     db = SessionLocal()
# #     try:
# #         manual_albums = db.query(Album).filter(Album.type == "manual").all()
# #         deleted = []
# #         for a in manual_albums:
# #             count = db.query(DBImage).filter(DBImage.album_id == a.id).count()
# #             if count == 0:
# #                 deleted.append(a.title)
# #                 db.delete(a)
# #         db.commit()
# #         return {"status": "done", "deleted": len(deleted), "titles": deleted}
# #     except Exception as e:
# #         db.rollback(); raise HTTPException(500, str(e))
# #     finally:
# #         db.close()


# # @app.post("/albums/{album_id}/rename")
# # def rename_album(album_id: int, title: str = Form(...), description: str = Form("")):
# #     """Rename an album and optionally update its description."""
# #     db = SessionLocal()
# #     try:
# #         album = db.query(Album).filter(Album.id == album_id).first()
# #         if not album:
# #             raise HTTPException(404, "Album not found")
# #         album.title = title.strip()
# #         if description.strip():
# #             album.description = description.strip()
# #         db.commit()
# #         return {"status": "renamed", "id": album_id, "title": album.title}
# #     except HTTPException: raise
# #     except Exception as e:
# #         db.rollback(); raise HTTPException(500, str(e))
# #     finally:
# #         db.close()


# # @app.post("/albums/{album_id}/add-images")
# # def add_images_to_album(album_id: int, image_ids: str = Form(...)):
# #     """Add a set of images (comma-separated IDs) to an existing album."""
# #     db = SessionLocal()
# #     try:
# #         album = db.query(Album).filter(Album.id == album_id).first()
# #         if not album:
# #             raise HTTPException(404, "Album not found")
# #         ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
# #         updated = _live(db).filter(DBImage.id.in_(ids)).update(
# #             {"album_id": album_id}, synchronize_session=False
# #         )
# #         # Refresh album dates
# #         imgs = _live(db).filter(DBImage.album_id == album_id).all()
# #         timestamps = [i.timestamp for i in imgs if i.timestamp]
# #         if timestamps:
# #             album.start_date = min(timestamps)
# #             album.end_date   = max(timestamps)
# #         db.commit()
# #         return {"status": "added", "album_id": album_id, "added": updated}
# #     except HTTPException: raise
# #     except Exception as e:
# #         db.rollback(); raise HTTPException(500, str(e))
# #     finally:
# #         db.close()


# # @app.post("/albums/{album_id}/remove-images")
# # def remove_images_from_album(album_id: int, image_ids: str = Form(...)):
# #     """Remove images from an album (images are kept, just unlinked)."""
# #     db = SessionLocal()
# #     try:
# #         ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
# #         _live(db).filter(DBImage.id.in_(ids), DBImage.album_id == album_id).update(
# #             {"album_id": None}, synchronize_session=False
# #         )
# #         db.commit()
# #         return {"status": "removed", "album_id": album_id}
# #     except Exception as e:
# #         db.rollback(); raise HTTPException(500, str(e))
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # DUPLICATES / EXPLORE / STATS
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.get("/duplicates")
# # def get_duplicates():
# #     db = SessionLocal()
# #     try:
# #         all_images = _live(db).all()
# #         if not all_images:
# #             return {"status": "found", "duplicate_groups": [], "total_groups": 0}
# #         groups    = duplicate_engine.find_duplicates_fast(all_images, hamming_threshold=5)
# #         formatted = [{"count": g["count"], "total_size": g["total_size"], "images": [{"id": img["id"], "filename": _img_url(img["filename"]), "thumbnail": _img_url(img["filename"]), "size": img["size"]} for img in g["images"]]} for g in groups]
# #         return {"status": "found", "duplicate_groups": formatted, "total_groups": len(formatted)}
# #     except Exception as e:
# #         logger.error(f"Duplicates error: {e}", exc_info=True)
# #         return {"status": "error", "message": str(e), "duplicate_groups": [], "total_groups": 0}
# #     finally:
# #         db.close()

# # @app.get("/explore/random")
# # def explore_random(count: int = Query(12)):
# #     import random as _random
# #     db = SessionLocal()
# #     try:
# #         all_ids = [row[0] for row in _live(db).with_entities(DBImage.id).all()]
# #         if not all_ids:
# #             return {"status": "not_found", "results": []}
# #         sampled = _random.sample(all_ids, min(count, len(all_ids)))
# #         imgs    = db.query(DBImage).filter(DBImage.id.in_(sampled)).all()
# #         return {"status": "found", "count": len(imgs), "results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None, "person_count": img.person_count or 0} for img in imgs]}
# #     finally:
# #         db.close()

# # @app.get("/stats")
# # def get_stats():
# #     import collections
# #     db = SessionLocal()
# #     try:
# #         total_images    = db.query(DBImage).count()
# #         total_faces     = db.query(DBFace).count()
# #         total_people    = db.query(Person).count()
# #         total_albums    = db.query(Album).count()
# #         total_favorites = _live(db).filter(DBImage.is_favorite == True).count()
# #         total_trashed   = db.query(DBImage).filter(DBImage.is_trashed == True).count()
# #         indexed         = search_engine.index.ntotal if search_engine.index else 0
# #         # AI object-detection labels (deduplicated per image)
# #         ai_counter = collections.Counter()
# #         for row in db.query(DBImage.scene_label).filter(DBImage.scene_label != None).all():
# #             seen_in_image = set()
# #             for tag in (row[0] or "").split(","):
# #                 t = tag.strip().lower()
# #                 if t and t not in seen_in_image:
# #                     ai_counter[t] += 1
# #                     seen_in_image.add(t)
# #         top_tags = [{"tag": t, "count": c} for t, c in ai_counter.most_common(10)]

# #         # User-defined tags
# #         user_tag_counter = collections.Counter()
# #         for row in db.execute(text("SELECT user_tags FROM images WHERE user_tags IS NOT NULL AND user_tags != '[]' AND (is_trashed IS NULL OR is_trashed=0)")):
# #             try:
# #                 import json as _j
# #                 for t in _j.loads(row[0]): user_tag_counter[t] += 1
# #             except Exception: pass
# #         top_user_tags = [{"tag": t, "count": c} for t, c in user_tag_counter.most_common(20)]
# #         COLOR_BUCKETS = {"red": (220,50,50), "orange": (230,120,40), "yellow": (220,210,50), "green": (50,160,50), "blue": (50,80,220), "purple": (120,50,180), "pink": (230,130,160), "white": (240,240,240), "black": (20,20,20), "gray": (128,128,128), "brown": (140,90,50)}
# #         color_dist = {k: 0 for k in COLOR_BUCKETS}
# #         for img in _live(db).filter(DBImage.avg_r != None).all():
# #             img_rgb = np.array([img.avg_r or 0, img.avg_g or 0, img.avg_b or 0], dtype=np.float32)
# #             best_bucket = min(COLOR_BUCKETS, key=lambda k: np.linalg.norm(img_rgb - np.array(COLOR_BUCKETS[k], dtype=np.float32)))
# #             color_dist[best_bucket] += 1
# #         return {"total_images": total_images, "total_faces": total_faces, "total_people": total_people, "total_albums": total_albums, "total_favorites": total_favorites, "total_trashed": total_trashed, "indexed_vectors": indexed, "top_tags": top_tags, "top_user_tags": top_user_tags, "color_distribution": sorted([{"color": k, "count": v} for k, v in color_dist.items() if v > 0], key=lambda x: x["count"], reverse=True)}
# #     finally:
# #         db.close()


# # # ── Common non-name words returned by BLIP when it doesn't know the name ───
# # _BLIP_NON_NAMES = {
# #     # articles / pronouns / common words
# #     "a", "an", "the", "it", "he", "she", "they", "him", "her", "them",
# #     "his", "hers", "their", "we", "our", "you", "your", "my", "i",
# #     # question words (critical — BLIP echoes these back)
# #     "what", "who", "which", "where", "when", "how", "why", "is", "are",
# #     "was", "were", "be", "been", "being", "do", "does", "did", "has",
# #     "have", "had", "will", "would", "could", "should", "may", "might",
# #     "can", "name", "named", "called", "known",
# #     # generic person descriptions
# #     "man", "woman", "person", "people", "boy", "girl", "child", "adult",
# #     "human", "face", "faces", "subject", "individual", "individuals",
# #     "unknown", "someone", "somebody", "anyone", "nobody", "one",
# #     # job titles / roles (BLIP often returns these)
# #     "celebrity", "actor", "actress", "model", "player", "star", "singer",
# #     "politician", "athlete", "musician", "artist", "president", "director",
# #     "manager", "leader", "officer", "official", "member", "representative",
# #     # visual/image words (BLIP captions)
# #     "image", "photo", "picture", "figure", "portrait", "photograph",
# #     "this", "that", "these", "those", "here", "there",
# #     # caption filler words BLIP uses
# #     "main", "shows", "show", "showing", "appears", "appear", "looking",
# #     "wearing", "holding", "standing", "sitting", "smiling",
# #     "in", "on", "at", "of", "for", "with", "from", "by", "to",
# #     "and", "or", "but", "not", "also", "as", "so", "then", "than",
# #     "black", "white", "two", "three", "four", "five", "several", "many",
# #     # Sports context — appear on jerseys, banners, scoreboards
# #     "india", "team", "cup", "match", "captain", "wicket", "wizard",
# #     "champion", "cricket", "football", "sport", "league", "series",
# #     "trophy", "test", "ipl", "bcci", "odi", "t20", "final", "world",
# #     "semi", "quarter", "squad", "player", "coach", "umpire", "referee",
# #     # Brand/sponsor words (F1, cricket, general)
# #     "petronas", "stake", "marlboro", "ferrari", "mercedes", "mclaren",
# #     "alpine", "williams", "haas", "redbull", "birla", "estates", "bank",
# #     "mutual", "fund", "limited", "pvt", "company", "corp", "inc", "ltd",
# #     "group", "holding", "sponsor", "official", "partner",
# #     # Common OCR fragments from Indian sports context
# #     "ndia", "iindia", "ldcup", "ineqs", "eironas", "hak", "zon",
# #     # Titles / honorifics misread as names
# #     "master", "mister", "miss", "mrs", "sir", "dr", "prof", "mr",
# #     # Generic English words that look capitalised in headlines
# #     "new", "old", "big", "great", "good", "best", "top", "live",
# #     "real", "true", "blue", "gold", "king", "queen", "super", "ultra",
# # }

# # # ── Patterns that definitively mark a word as NOT a person name ─────────────
# # def _is_valid_name_word(w: str, allow_short: bool = False) -> bool:
# #     """
# #     Return True only if `w` looks like part of a real person's name.
# #     Rejects: ALL-CAPS brands, too-short initials, non-alpha, known non-names.
# #     """
# #     w_clean = w.rstrip(".,!?;:'\"")
# #     if not w_clean:
# #         return False
# #     # Must be purely alphabetic (no digits, no hyphens, no underscores)
# #     if not w_clean.isalpha():
# #         return False
# #     # ALL-CAPS words are brand names / acronyms / jersey text, not person names
# #     if w_clean == w_clean.upper() and len(w_clean) > 2:
# #         return False
# #     # Minimum length — real given names are rarely < 3 chars in isolation
# #     min_len = 3 if allow_short else 4
# #     if len(w_clean) < min_len:
# #         return False
# #     # Must start with uppercase (proper noun)
# #     if not w_clean[0].isupper():
# #         return False
# #     # Must not be a known non-name
# #     if w_clean.lower() in _BLIP_NON_NAMES:
# #         return False
# #     # Must contain at least one vowel (pure consonant strings = OCR garbage)
# #     if not any(c in "aeiouAEIOU" for c in w_clean):
# #         return False
# #     return True


# # def _extract_name_from_vqa(vqa_text: str) -> str:
# #     """
# #     Parse a BLIP VQA answer to extract a plausible person name.
# #     Very strict — only returns something if it looks like a real proper name.
# #     Returns empty string if uncertain.
# #     """
# #     if not vqa_text:
# #         return ""
# #     text = vqa_text.strip()

# #     # Strip common prefixes BLIP adds
# #     for prefix in [
# #         "the person is ", "this is ", "it is ", "that is ",
# #         "his name is ", "her name is ", "the name is ",
# #         "the man is ", "the woman is ", "i think it is ",
# #         "i think it's ", "i believe it is ", "appears to be ",
# #         "looks like ", "this appears to be ", "the answer is ",
# #     ]:
# #         if text.lower().startswith(prefix):
# #             text = text[len(prefix):]

# #     # Take just the first 1-2 words
# #     words = text.split()[:2]
# #     if not words:
# #         return ""

# #     # Validate each word — allow short (3-char) first names only in 2-word combos
# #     if len(words) == 2:
# #         w1_ok = _is_valid_name_word(words[0], allow_short=True)
# #         w2_ok = _is_valid_name_word(words[1], allow_short=False)
# #         if w1_ok and w2_ok:
# #             name = f"{words[0].rstrip('.,!?;:')} {words[1].rstrip('.,!?;:')}"
# #             return name if 4 <= len(name) <= 35 else ""
# #         if w1_ok:
# #             w = words[0].rstrip(".,!?;:")
# #             return w if 4 <= len(w) <= 35 else ""
# #         return ""
# #     else:
# #         w = words[0].rstrip(".,!?;:")
# #         if _is_valid_name_word(w, allow_short=False):
# #             return w if 4 <= len(w) <= 35 else ""
# #         return ""


# # def _auto_name_people(db, person_map: dict):
# #     """
# #     For each newly created person cluster, try to auto-name them.
# #     Strategy (strict):
# #       1. VQA "person" answers get highest weight (3 pts each) — most reliable
# #       2. OCR proper nouns that appear in MULTIPLE images of the same cluster get 1 pt each
# #          (single-image OCR hits are often jersey text / background banners → ignored)
# #       3. Only assign a name if the top candidate has >= 50% of photos voting for it
# #          AND if it's a VQA-sourced name, OR >= 2 images (to prevent single-image OCR garbage)
# #     """
# #     import json as _jj
# #     from collections import Counter

# #     for label, person_id in person_map.items():
# #         person = db.query(Person).filter(Person.id == person_id).first()
# #         if not person:
# #             continue
# #         # Only auto-name if still using the default label
# #         if not person.name.startswith("Person "):
# #             continue

# #         face_records = db.query(DBFace).filter(DBFace.person_id == person_id).all()
# #         img_ids = list({f.image_id for f in face_records if f.image_id})
# #         if not img_ids:
# #             continue

# #         images = db.query(DBImage).filter(DBImage.id.in_(img_ids)).all()
# #         n_images = len(images)

# #         # Separate counters: VQA (trusted) vs OCR (needs cross-image confirmation)
# #         vqa_counter = Counter()
# #         ocr_counter = Counter()   # counts how many IMAGES contain each OCR name

# #         for img in images:
# #             # ── VQA: most reliable, use directly ────────────────────────────
# #             if img.caption_vqa:
# #                 try:
# #                     vqa = _jj.loads(img.caption_vqa)
# #                     vqa_name = _extract_name_from_vqa(vqa.get("person", ""))
# #                     if vqa_name:
# #                         vqa_counter[vqa_name] += 1
# #                 except Exception:
# #                     pass

# #             # ── OCR: scan for proper-noun pairs and single words ─────────────
# #             # Only count a name ONCE per image (to prevent a single jersey from
# #             # winning just because the text appears 10 times in one photo).
# #             ocr = img.ocr_text_enhanced or ""
# #             if ocr:
# #                 seen_in_this_image = set()
# #                 words = ocr.split()
# #                 for i, w in enumerate(words):
# #                     # Try two-word name first
# #                     if i + 1 < len(words):
# #                         w2 = words[i + 1].rstrip(".,!?;:'\"")
# #                         w1 = w.rstrip(".,!?;:'\"")
# #                         if (_is_valid_name_word(w1, allow_short=True) and
# #                                 _is_valid_name_word(w2, allow_short=False)):
# #                             two_word = f"{w1} {w2}"
# #                             if two_word not in seen_in_this_image:
# #                                 seen_in_this_image.add(two_word)
# #                                 ocr_counter[two_word] += 1

# #                     # Single word fallback
# #                     w1 = w.rstrip(".,!?;:'\"")
# #                     if (_is_valid_name_word(w1, allow_short=False) and
# #                             w1 not in seen_in_this_image):
# #                         seen_in_this_image.add(w1)
# #                         ocr_counter[w1] += 1

# #         # ── Decision logic ────────────────────────────────────────────────────
# #         best_name = None
# #         best_score = 0

# #         # VQA wins if it appears in >= 25% of images (e.g. 1 out of 4)
# #         if vqa_counter:
# #             top_vqa, top_vqa_count = vqa_counter.most_common(1)[0]
# #             min_vqa = max(1, n_images * 0.25)
# #             if top_vqa_count >= min_vqa:
# #                 best_name  = top_vqa
# #                 best_score = top_vqa_count * 3  # weighted

# #         # OCR only wins if:
# #         # (a) NO VQA name found, AND
# #         # (b) name appears in >= 2 different images (cross-image confirmation), AND
# #         # (c) appears in >= 40% of the cluster's images
# #         if not best_name and ocr_counter:
# #             top_ocr, top_ocr_count = ocr_counter.most_common(1)[0]
# #             min_ocr_images = max(2, n_images * 0.40)
# #             if top_ocr_count >= min_ocr_images:
# #                 best_name  = top_ocr
# #                 best_score = top_ocr_count

# #         if best_name:
# #             person.name = best_name
# #             logger.info(f"🏷️  Auto-named Person {person_id} → '{best_name}' "
# #                         f"(score={best_score}, vqa={dict(vqa_counter.most_common(3))}, "
# #                         f"n_images={n_images})")
# #         else:
# #             top_vqa_str = str(vqa_counter.most_common(1)) if vqa_counter else "none"
# #             top_ocr_str = str(ocr_counter.most_common(3)) if ocr_counter else "none"
# #             logger.info(f"⚠️  No confident name for Person {person_id} "
# #                         f"(n_images={n_images}, vqa={top_vqa_str}, top_ocr={top_ocr_str})")


# # # ────────────────────────────────────────────────────────────────────────────
# # # RECLUSTER
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.post("/recluster")

# # def recluster():
# #     db = SessionLocal()
# #     try:
# #         logger.info("🔄 Recluster: saving manually assigned names before reset…")

# #         # ── Save manually named people BEFORE wiping everything ─────────────
# #         # For each person with a non-default name, remember which face embeddings
# #         # (as indices into the face table) belonged to them.
# #         saved_names = {}  # face_id_set -> name
# #         for person in db.query(Person).all():
# #             pname = (person.name or "").strip()
# #             # Only keep explicitly renamed people (skip "Person N" defaults)
# #             if re.match(r"^person\s+\d+$", pname.lower()):
# #                 continue
# #             if not pname or pname.lower() in ("unknown", ""):
# #                 continue
# #             face_ids = frozenset(
# #                 f.id for f in db.query(DBFace).filter(DBFace.person_id == person.id).all()
# #             )
# #             if face_ids:
# #                 saved_names[face_ids] = pname
# #                 logger.info(f"💾 Saved name '{pname}' ({len(face_ids)} faces)")

# #         logger.info(f"🔄 Recluster: clearing old assignments…")
# #         db.query(DBFace).update({"person_id": None})
# #         db.query(Person).delete()

# #         # FIX: Only unset album_id for images in EVENT albums.
# #         # Manual albums created by the user must not be wiped.
# #         event_album_ids = [
# #             a.id for a in db.query(Album).filter(Album.type == "event").all()
# #         ]
# #         if event_album_ids:
# #             db.query(DBImage).filter(
# #                 DBImage.album_id.in_(event_album_ids)
# #             ).update({"album_id": None}, synchronize_session=False)
# #         db.query(Album).filter(Album.type == "event").delete()
# #         db.commit()

# #         # ── Load face records with image dimensions for size-filtering ────────
# #         # Background faces (tiny, far-away people) corrupt clusters — the same
# #         # main person ends up split into multiple clusters because their face
# #         # sometimes appears blurry/small in group shots.
# #         # Strategy: for each image, only cluster faces that are "prominent":
# #         #   - face area >= MIN_FACE_RATIO of image area (ignores tiny bystanders)
# #         #   - if multiple faces pass the threshold, keep top MAX_FACES_PER_IMAGE
# #         #     by size (supports group photos with 2-3 main subjects)
# #         MIN_FACE_RATIO   = 0.005   # face must cover >= 0.5% of image area
# #                                     # 1920×1080 image → face must be >= ~100×100 px
# #         MAX_FACES_PER_IMAGE = 4    # keep at most 4 largest faces per image

# #         # Build image dimension lookup: image_id → (width, height)
# #         img_dims = {}
# #         for img in db.query(DBImage.id, DBImage.width, DBImage.height).all():
# #             if img.width and img.height:
# #                 img_dims[img.id] = (img.width, img.height)

# #         face_records            = db.query(DBFace).filter(DBFace.face_embedding != None).all()
# #         embeddings, valid_faces = [], []
# #         skipped = 0
# #         bg_filtered = 0

# #         # Group faces by image to allow per-image top-N selection
# #         from collections import defaultdict
# #         import json as _jfilt
# #         faces_by_image = defaultdict(list)
# #         for fr in face_records:
# #             if not fr.image_id:
# #                 continue
# #             try:
# #                 emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
# #                 if emb.shape[0] != 512:
# #                     skipped += 1
# #                     continue
# #                 # Parse bbox to get face area
# #                 face_area_ratio = 1.0  # default: include if no bbox/dims available
# #                 if fr.bbox and fr.image_id in img_dims:
# #                     try:
# #                         bbox = _jfilt.loads(fr.bbox)  # [x1, y1, x2, y2]
# #                         x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
# #                         face_area = max(0, x2 - x1) * max(0, y2 - y1)
# #                         img_w, img_h = img_dims[fr.image_id]
# #                         img_area = img_w * img_h
# #                         face_area_ratio = face_area / img_area if img_area > 0 else 1.0
# #                     except Exception:
# #                         face_area_ratio = 1.0  # parse failed → include by default
# #                 faces_by_image[fr.image_id].append((face_area_ratio, fr, emb))
# #             except Exception as e:
# #                 logger.warning(f"Bad embedding face {fr.id}: {e}")
# #                 skipped += 1

# #         # For each image: filter by MIN_FACE_RATIO, then keep top MAX_FACES_PER_IMAGE
# #         for image_id, face_list in faces_by_image.items():
# #             # Sort by area ratio descending (largest face first)
# #             face_list.sort(key=lambda x: x[0], reverse=True)

# #             # Apply area filter — but always keep at least 1 face per image
# #             # (portrait shots where the face fills the frame pass trivially)
# #             passed = [item for item in face_list if item[0] >= MIN_FACE_RATIO]
# #             if not passed:
# #                 # All faces were small — keep the single largest one anyway
# #                 # (could be a small portrait photo shot from far away)
# #                 passed = face_list[:1]
# #                 logger.debug(f"🔍 Image {image_id}: all faces small, keeping largest "
# #                              f"(ratio={face_list[0][0]:.4f})")

# #             # Cap at MAX_FACES_PER_IMAGE
# #             kept = passed[:MAX_FACES_PER_IMAGE]
# #             dropped = len(face_list) - len(kept)
# #             if dropped > 0:
# #                 bg_filtered += dropped
# #                 logger.debug(f"🔍 Image {image_id}: kept {len(kept)}/{len(face_list)} faces "
# #                              f"(dropped {dropped} background faces, "
# #                              f"min_ratio={kept[-1][0]:.4f})")

# #             for _, fr, emb in kept:
# #                 embeddings.append(emb)
# #                 valid_faces.append(fr)

# #         logger.info(f"👥 {len(embeddings)} prominent face embeddings "
# #                     f"({skipped} invalid, {bg_filtered} background faces filtered out)")
# #         people_count = 0
# #         if embeddings:
# #             labels     = face_engine.cluster_faces(embeddings)
# #             person_map = {}
# #             for i, label in enumerate(labels):
# #                 if label == -1: continue
# #                 if label not in person_map:
# #                     p = Person(name=f"Person {label + 1}")
# #                     db.add(p); db.flush()
# #                     person_map[label] = p.id; people_count += 1
# #                 valid_faces[i].person_id = person_map[label]
# #             db.commit()
# #             logger.info(f"✅ {people_count} people created")

# #             # ── Restore manually assigned names to matching new clusters ────
# #             if saved_names:
# #                 restored = 0
# #                 # Build map: face_id -> new person_id
# #                 face_to_person = {}
# #                 for face in db.query(DBFace).filter(DBFace.person_id != None).all():
# #                     face_to_person[face.id] = face.person_id

# #                 for old_face_ids, saved_name in saved_names.items():
# #                     # Find which new person cluster has the most overlap with old faces
# #                     overlap_count = {}
# #                     for fid in old_face_ids:
# #                         pid = face_to_person.get(fid)
# #                         if pid:
# #                             overlap_count[pid] = overlap_count.get(pid, 0) + 1

# #                     if not overlap_count:
# #                         continue
# #                     # Best matching cluster
# #                     best_pid = max(overlap_count, key=overlap_count.get)
# #                     best_overlap = overlap_count[best_pid]
# #                     # Only restore if at least 30% of faces match
# #                     min_match = max(1, len(old_face_ids) * 0.30)
# #                     if best_overlap >= min_match:
# #                         person = db.query(Person).filter(Person.id == best_pid).first()
# #                         if person and re.match(r"^person\s+\d+$", (person.name or "").lower()):
# #                             person.name = saved_name
# #                             restored += 1
# #                             logger.info(f"✅ Restored name '{saved_name}' → Person {best_pid} ({best_overlap}/{len(old_face_ids)} faces matched)")
# #                 db.commit()
# #                 logger.info(f"🏷️  Restored {restored}/{len(saved_names)} manually assigned names")

# #             # Auto-name remaining unnamed clusters from BLIP captions / VQA
# #             _auto_name_people(db, person_map)
# #             db.commit()
# #         all_images   = db.query(DBImage).all()
# #         albums_count = 0
# #         if all_images:
# #             metadata = [{"id": img.id, "lat": img.lat or 0.0, "lon": img.lon or 0.0, "timestamp": img.timestamp} for img in all_images if img.timestamp]
# #             if metadata:
# #                 album_labels = clustering_engine.detect_events(metadata)
# #                 album_map    = {}
# #                 for i, label in enumerate(album_labels):
# #                     if label == -1: continue
# #                     if label not in album_map:
# #                         cluster_meta = [metadata[j] for j, l in enumerate(album_labels) if l == label]
# #                         ts_list      = [m["timestamp"] for m in cluster_meta if m["timestamp"]]
# #                         start_d      = min(ts_list) if ts_list else None
# #                         end_d        = max(ts_list) if ts_list else None
# #                         if start_d:
# #                             if end_d and end_d.date() != start_d.date():
# #                                 # Multi-day: "Mar 4 – 7, 2026"
# #                                 if end_d.month == start_d.month:
# #                                     title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%d, %Y')}"
# #                                 else:
# #                                     title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%b %d, %Y')}"
# #                             else:
# #                                 # Single day: "Mar 4, 2026"
# #                                 title = start_d.strftime("%b %d, %Y")
# #                         else:
# #                             title = f"Event {label + 1}"
# #                         new_album = Album(title=title, type="event", start_date=start_d, end_date=end_d)
# #                         db.add(new_album); db.flush()
# #                         album_map[label] = new_album.id; albums_count += 1
# #                     db.query(DBImage).filter(DBImage.id == metadata[i]["id"]).update({"album_id": album_map[label]})
# #                 db.commit()
# #                 logger.info(f"✅ {albums_count} albums created")
# #         return {"status": "done", "people": people_count, "albums": albums_count}
# #     except Exception as e:
# #         db.rollback()
# #         logger.error(f"❌ Recluster failed: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=str(e))
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # UPLOAD
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.post("/upload")
# # async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
# #     ext = os.path.splitext(file.filename)[1].lower()
# #     if ext not in [".jpg", ".jpeg", ".png"]:
# #         raise HTTPException(status_code=400, detail="Only JPG and PNG supported.")
# #     filename  = f"{uuid.uuid4()}{ext}"
# #     file_path = os.path.join(IMAGE_DIR, filename)
# #     db        = SessionLocal()
# #     try:
# #         with open(file_path, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)
# #         from PIL import Image as PILImage
# #         width = height = None
# #         avg_r = avg_g = avg_b = 0.0
# #         try:
# #             img_pil       = PILImage.open(file_path).convert("RGB")
# #             width, height = img_pil.size
# #             arr = np.array(img_pil)
# #             # Use dominant color (most common cluster) instead of plain average
# #             # Resize to 64x64 for speed, then find the peak of the color histogram
# #             import colorsys as _cs
# #             small = np.array(img_pil.resize((64, 64)), dtype=np.float32)
# #             r_vals = small[:,:,0].flatten()
# #             g_vals = small[:,:,1].flatten()
# #             b_vals = small[:,:,2].flatten()
# #             # Weight pixels by their saturation (ignore grey/white/black background)
# #             weights = []
# #             for ri, gi, bi in zip(r_vals, g_vals, b_vals):
# #                 _, s, v = _cs.rgb_to_hsv(ri/255, gi/255, bi/255)
# #                 weights.append(s * v + 0.01)  # saturated bright pixels count more
# #             weights = np.array(weights)
# #             total_w = weights.sum()
# #             avg_r = float(np.dot(r_vals, weights) / total_w)
# #             avg_g = float(np.dot(g_vals, weights) / total_w)
# #             avg_b = float(np.dot(b_vals, weights) / total_w)
# #         except Exception:
# #             pass
# #         clip_emb = None
# #         try:
# #             clip_emb = search_engine.get_image_embedding(file_path)
# #         except Exception:
# #             pass
# #         scene_label  = ""
# #         person_count = 0
# #         try:
# #             objects      = detector_engine.detect_objects(file_path, threshold=0.5)
# #             scene_label  = ", ".join(objects) if objects else ""
# #             person_count = detector_engine.detect_persons(file_path)
# #         except Exception:
# #             pass
# #         img_record = DBImage(filename=filename, original_path=file_path, timestamp=datetime_module.datetime.now(), width=width, height=height, avg_r=avg_r, avg_g=avg_g, avg_b=avg_b, scene_label=scene_label, person_count=person_count, ocr_text_enhanced="", ocr_keywords="[]", ocr_confidence=0.0, detected_language="en", caption_short="", caption_detailed="", caption_vqa="{}", quality_score=0.0, quality_level="Processing...", sharpness=0.0, exposure=0.0, contrast=0.0, composition=0.0, emotion_data="[]", dominant_emotion="neutral", face_emotion_count=0, aesthetic_score=0.0, aesthetic_rating="Processing...")
# #         db.add(img_record); db.flush()
# #         if clip_emb is not None:
# #             try:
# #                 if search_engine.index is None:
# #                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
# #                 new_vec = clip_emb.reshape(1, -1).astype("float32")
# #                 faiss.normalize_L2(new_vec)
# #                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id], dtype="int64"))
# #                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
# #             except Exception as e:
# #                 logger.warning(f"FAISS update failed: {e}")
# #         face_count = 0
# #         try:
# #             faces = face_engine.detect_faces(file_path)
# #             if faces and width and height:
# #                 img_area = width * height
# #                 # Sort faces by area descending (largest = most prominent first)
# #                 def _face_area(f):
# #                     b = f["bbox"]  # [x1, y1, x2, y2]
# #                     return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
# #                 faces_sorted = sorted(faces, key=_face_area, reverse=True)
# #                 # Keep only prominent faces (>= 0.5% of image area), max 4
# #                 prominent = [f for f in faces_sorted
# #                              if _face_area(f) / img_area >= 0.005]
# #                 if not prominent:
# #                     prominent = faces_sorted[:1]  # always keep largest face
# #                 faces = prominent[:4]
# #             for face in faces:
# #                 emb = face["embedding"].astype(np.float32)
# #                 db.add(DBFace(image_id=img_record.id, bbox=_json.dumps(face["bbox"]), face_embedding=emb.tobytes()))
# #                 face_count += 1
# #         except Exception as e:
# #             logger.warning(f"Face detection failed: {e}")
# #         db.commit()
# #         image_id = img_record.id
# #         if background_tasks is not None:
# #             background_tasks.add_task(_enrich_image, image_id, file_path)
# #             should_trigger_recluster(background_tasks)
# #         else:
# #             _enrich_image(image_id, file_path)
# #         logger.info(f"✅ Upload done: {filename} (id={image_id})")
# #         return {"status": "success", "id": image_id, "filename": filename, "person_count": person_count, "face_count": face_count, "quality_score": 0.0, "quality_level": "Processing...", "caption": "", "dominant_emotion": "neutral", "emotion_count": 0, "aesthetic_score": 0.0, "note": "ML enrichment running in background"}
# #     except Exception as e:
# #         db.rollback()
# #         if os.path.exists(file_path): os.remove(file_path)
# #         logger.error(f"Upload failed: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=str(e))
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # REPROCESS
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.post("/reprocess_images")
# # async def reprocess_images():
# #     db = SessionLocal()
# #     processed, failed = 0, 0
# #     try:
# #         images = db.query(DBImage).filter((DBImage.caption_short == None) | (DBImage.caption_short == "") | (DBImage.quality_level == "Processing...")).all()
# #         if not images:
# #             return {"success": True, "processed": 0, "failed": 0}
# #         for img in images:
# #             try:
# #                 if not img.original_path or not os.path.exists(img.original_path):
# #                     failed += 1; continue
# #                 _enrich_image(img.id, img.original_path); processed += 1
# #             except Exception:
# #                 failed += 1
# #         return {"success": True, "processed": processed, "failed": failed}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# #     finally:
# #         db.close()


# # # ────────────────────────────────────────────────────────────────────────────
# # # RECAPTION — re-run BLIP on images with missing or poor captions
# # # ────────────────────────────────────────────────────────────────────────────
# # @app.post("/recaption")
# # async def recaption_images(background_tasks: BackgroundTasks, force_all: bool = False):
# #     """
# #     Re-run BLIP captioning on images that:
# #     - Have no caption (caption_short is NULL or empty)
# #     - Have a placeholder/error caption
# #     - force_all=True: re-caption every image (useful after switching BLIP model)

# #     Runs in the background so the API returns immediately.
# #     Progress can be checked via GET /recaption/status
# #     """
# #     db = SessionLocal()
# #     try:
# #         if force_all:
# #             images = _live(db).all()
# #         else:
# #             # Only images missing captions
# #             images = _live(db).filter(
# #                 (DBImage.caption_short == None) |
# #                 (DBImage.caption_short == "") |
# #                 (DBImage.caption_short == "Processing...")
# #             ).all()

# #         pairs = [
# #             (img.id, img.original_path or os.path.join(IMAGE_DIR, img.filename))
# #             for img in images
# #         ]
# #         # Filter to files that actually exist
# #         pairs = [(iid, fp) for iid, fp in pairs if os.path.exists(fp)]
# #     finally:
# #         db.close()

# #     if not pairs:
# #         return {"status": "nothing_to_do", "count": 0,
# #                 "message": "All images already have captions. Use force_all=true to re-caption everything."}

# #     def _run_recaption(image_pairs):
# #         done = 0
# #         for img_id, fpath in image_pairs:
# #             try:
# #                 db2 = SessionLocal()
# #                 try:
# #                     caption_short    = captioning_engine.generate_caption(fpath, max_length=20)
# #                     caption_detailed = captioning_engine.generate_caption(fpath, max_length=60)
# #                     vqa_subject = captioning_engine.answer_visual_question(
# #                         fpath, "What is the main subject in this image?"
# #                     )
# #                     # Check if this image has people for VQA person question
# #                     img_chk = db2.query(DBImage).filter(DBImage.id == img_id).first()
# #                     vqa_person = ""
# #                     if img_chk and (img_chk.person_count or 0) > 0:
# #                         vqa_person = captioning_engine.answer_visual_question(fpath, "who is this?")
# #                         if vqa_person and vqa_person.lower().strip() in {
# #                             "man","woman","person","a man","a woman","a person","boy","girl",
# #                             "child","human","unknown","celebrity","actor","actress","no","yes",
# #                             "none","i don't know",
# #                         }:
# #                             vqa_person = ""

# #                     db2.query(DBImage).filter(DBImage.id == img_id).update({
# #                         "caption_short":    caption_short,
# #                         "caption_detailed": caption_detailed,
# #                         "caption_vqa":      _json.dumps({
# #                             "subject": vqa_subject,
# #                             "person":  vqa_person,
# #                         } if (vqa_subject or vqa_person) else {}),
# #                         "caption_timestamp": datetime_module.datetime.now(),
# #                     }, synchronize_session=False)
# #                     db2.commit()
# #                     done += 1
# #                     logger.info(f"✅ Recaptioned [{img_id}]: {caption_short}")
# #                 finally:
# #                     db2.close()
# #             except Exception as e:
# #                 logger.warning(f"⚠️ Recaption failed [{img_id}]: {e}")
# #         logger.info(f"✅ Recaption complete: {done}/{len(image_pairs)} images")

# #     background_tasks.add_task(_run_recaption, pairs)
# #     return {
# #         "status": "started",
# #         "count": len(pairs),
# #         "message": f"Re-captioning {len(pairs)} images in background. "
# #                    f"{'(all images)' if force_all else '(missing captions only)'}",
# #     }


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
# import json as _json
# from enhanced_ocr_engine import ocr_engine
# from image_captioning_engine import captioning_engine
# from quality_emotion_aesthetic_engines import (
#     image_quality,
#     emotion_detection,
#     aesthetic_scoring
# )

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
# from fastapi.responses import FileResponse
# from sqlalchemy import text
# import re
# import json
# from api_endpoints import router as dl_router
# from voice_route import router as voice_router
# from features_router import router as feat_router, ensure_extra_columns   # ← only import here

# _BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
# IMAGE_DIR        = os.path.normpath(os.path.join(_BASE_DIR, "..", "data", "images"))
# FAISS_INDEX_PATH = os.path.normpath(os.path.join(_BASE_DIR, "..", "data", "index.faiss"))

# FACE_MATCH_THRESHOLD  = float(os.environ.get("FACE_MATCH_THRESHOLD",  0.75))
# FACE_MATCH_NEIGHBORS  = int(os.environ.get("FACE_MATCH_NEIGHBORS",    5))
# FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
# RECLUSTER_ON_UPLOAD   = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
# RECLUSTER_BATCH_SIZE  = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

# CLIP_SCORE_MIN  = float(os.environ.get("CLIP_SCORE_MIN",  0.14))  # 0.14: catches football/soccer/cat/dog matches
# THRESHOLD_FLOOR = float(os.environ.get("THRESHOLD_FLOOR", 0.22))
# ADAPTIVE_RATIO  = float(os.environ.get("ADAPTIVE_RATIO",  0.92))
# FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.13))  # 0.13: matches CLIP threshold

# RECLUSTER_COUNTER_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
# RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 8.0))  # 8s: fire soon after upload
# recluster_last_triggered = None

# COLOR_SCORE_MAP = {
#     'red':    (1.0, 0,    0),    'blue':   (0,   0,    1.0),
#     'green':  (0,   1.0,  0),    'yellow': (1.0, 1.0,  0),
#     'orange': (1.0, 0.5,  0),    'purple': (0.5, 0,    0.5),
#     'pink':   (1.0, 0.75, 0.8),  'black':  (0,   0,    0),
#     'white':  (1,   1,    1),    'gray':   (0.5, 0.5,  0.5),
#     'brown':  (0.6, 0.4,  0.2),
# }


# # ── Emotion emojis → search terms (intercept before CLIP) ───────────────────
# EMOTION_EMOJI_MAP = {
#     "😊": "happy",  "😄": "happy",  "😁": "happy",  "🙂": "happy",  "😀": "happy",
#     "😃": "happy",  "🤗": "happy",  "😆": "happy",  "😂": "happy",  "🥰": "happy",
#     "😍": "happy",  "😎": "happy",  "🥳": "happy",  "😇": "happy",
#     "😢": "sad",    "😭": "sad",    "😔": "sad",    "😟": "sad",    "🥺": "sad",
#     "😞": "sad",    "😿": "sad",    "💔": "sad",
#     "😠": "angry",  "😡": "angry",  "🤬": "angry",  "😤": "angry",  "👿": "angry",
#     "😲": "surprised", "😮": "surprised", "🤯": "surprised", "😱": "surprised",
#     "🤢": "disgusted", "🤮": "disgusted", "😷": "disgusted",
#     "😨": "fearful",   "😰": "fearful",   "😱": "fearful",   "😧": "fearful",
#     "😐": "neutral",   "😑": "neutral",   "🙄": "neutral",   "😶": "neutral",
# }

# # Extended scene emoji map (supplements search_engine.py EMOJI_MAP)
# SCENE_EMOJI_MAP = {
#     "🌅": "sunset golden hour", "🌃": "city night",   "🏕️": "camping outdoor",
#     "🌴": "tropical palm tree", "⛰️": "mountain",     "🌊": "ocean waves beach",
#     "🎊": "party celebration",  "👪": "family group",  "🤳": "selfie portrait",
#     "🐕": "dog",                "🐈": "cat",            "🦁": "lion animal",
#     "🍽️": "food meal",          "☕": "coffee",         "🍰": "cake dessert",
#     "🏋️": "gym exercise",       "🧘": "yoga meditation","🏊": "swimming pool",
#     "🌸": "flowers spring",     "🍂": "autumn fall leaves",
#     "🎸": "music guitar",       "📸": "camera photography",
#     "⚽": "soccer football sport", "🏀": "basketball sport",
#     "🏈": "american football sport", "🎾": "tennis sport",
#     "🏏": "cricket sport bat", "🏒": "hockey sport",
#     "⚾": "baseball sport",    "🥊": "boxing sport",
# }

# # Sport/activity keyword expansions (applied in _clean_query)
# SPORT_SYNONYMS = {
#     "football": "football soccer sport player",
#     "soccer":   "soccer football sport player",
#     "cricket":  "cricket sport bat player",
#     "basketball": "basketball sport player",
#     "tennis":   "tennis sport player",
#     "swimming": "swimming pool sport",
#     "running":  "running jogging sport",
#     "cycling":  "cycling bicycle sport",
#     "gym":      "gym workout exercise",
#     "yoga":     "yoga meditation exercise",
# }

# # ── Text keyword → emotion mapping ─────────────────────────────────────────
# EMOTION_KEYWORDS = {
#     "happy":     ["happy", "happiness", "smiling", "smile", "laughing", "laugh",
#                   "joy", "joyful", "cheerful", "glad", "delighted", "pleased"],
#     "sad":       ["sad", "sadness", "crying", "cry", "unhappy", "upset",
#                   "depressed", "gloomy", "sorrow", "sorrowful", "tears"],
#     "angry":     ["angry", "anger", "mad", "furious", "rage", "annoyed", "irritated"],
#     "surprised": ["surprised", "surprise", "shocked", "shock", "amazed", "astonished"],
#     "fearful":   ["fearful", "fear", "scared", "afraid", "frightened", "terrified"],
#     "disgusted": ["disgusted", "disgust", "disgusting"],
#     "neutral":   ["neutral", "expressionless", "blank"],
# }

# def _extract_emotion_from_query(query: str):
#     """Return emotion name if query contains an emotion emoji or keyword, else None."""
#     # Check emojis first
#     for emoji, emo in EMOTION_EMOJI_MAP.items():
#         if emoji in query:
#             return emo
#     # Check text keywords
#     q_lower = query.lower()
#     for emo, keywords in EMOTION_KEYWORDS.items():
#         if any(kw in q_lower for kw in keywords):
#             return emo
#     return None

# def _expand_query_emojis(query: str) -> str:
#     """Replace scene emojis with text equivalents for better CLIP embedding."""
#     result = query
#     for emoji, text in SCENE_EMOJI_MAP.items():
#         result = result.replace(emoji, " " + text + " ")
#     return " ".join(result.split())


# def _live(db):
#     return db.query(DBImage).filter(
#         (DBImage.is_trashed == False) | (DBImage.is_trashed == None)
#     )


# def _img_url(filename: str) -> str:
#     if not filename:
#         return None
#     return os.path.basename(filename)


# def _user_tags(img) -> list:
#     """Safely decode the user_tags JSON column into a Python list."""
#     try:
#         v = getattr(img, 'user_tags', None)
#         return json.loads(v) if v and v not in ('[]', '') else []
#     except Exception:
#         return []


# def _clean_query(query: str) -> str:
#     import unicodedata, re as _re
#     processed = resolve_query(query)
#     result = []
#     for char in processed:
#         cp = ord(char)
#         if cp > 127:
#             cat = unicodedata.category(char)
#             if cat in ("So", "Sm", "Sk", "Mn"):
#                 try:
#                     name = unicodedata.name(char, "").lower()
#                     name = name.replace(" sign", "").replace(" symbol", "")
#                     name = _re.sub(r"\bwith\b.*", "", name).strip()
#                     result.append(" " + name + " ")
#                 except Exception:
#                     pass
#             else:
#                 result.append(char)
#         else:
#             result.append(char)
#     cleaned = _re.sub(r"\s+", " ", "".join(result)).strip()
#     return cleaned if cleaned else processed


# def _startup_fix_filenames():
#     db = SessionLocal()
#     try:
#         fixed = 0
#         rows = db.query(DBImage).filter(DBImage.filename.contains("/")).all()
#         for r in rows:
#             r.filename = os.path.basename(r.filename)
#             fixed += 1
#         if fixed:
#             db.commit()
#             logger.info(f"Startup fix: normalised {fixed} filenames in DB")
#     except Exception as e:
#         db.rollback()
#         logger.warning(f"Startup filename fix failed: {e}")
#     finally:
#         db.close()


# def _enrich_image(image_id: int, file_path: str):
#     db = SessionLocal()
#     try:
#         img_record = db.query(DBImage).filter(DBImage.id == image_id).first()
#         if not img_record:
#             return
#         try:
#             img_record.caption_short    = captioning_engine.generate_caption(file_path, max_length=20)
#             img_record.caption_detailed = captioning_engine.generate_caption(file_path, max_length=60)
#             vqa_subject = captioning_engine.answer_visual_question(file_path, "What is the main subject in this image?")
#             vqa_person  = ""
#             if img_record.person_count and img_record.person_count > 0:
#                 # Ask BLIP to describe the person — identity recognition
#                 # works best with very short questions
#                 vqa_person = captioning_engine.answer_visual_question(
#                     file_path, "who is this?"
#                 )
#                 # Discard generic answers immediately
#                 if vqa_person and vqa_person.lower().strip() in {
#                     "man", "woman", "person", "a man", "a woman", "a person",
#                     "boy", "girl", "child", "human", "unknown", "celebrity",
#                     "actor", "actress", "no", "yes", "none", "i don't know",
#                 }:
#                     vqa_person = ""
#             img_record.caption_vqa = _json.dumps({
#                 "subject": vqa_subject,
#                 "person":  vqa_person,
#             } if (vqa_subject or vqa_person) else {})
#             img_record.caption_timestamp = datetime_module.datetime.now()
#             logger.info(f"✅ Caption [{image_id}]: {img_record.caption_short}")
#         except Exception as e:
#             logger.warning(f"⚠️  Caption failed [{image_id}]: {e}")
#         try:
#             img_record.ocr_text_enhanced = ocr_engine.extract_text(file_path)
#             kw = ocr_engine.extract_text_with_confidence(file_path)
#             img_record.ocr_keywords   = _json.dumps([i["text"] for i in kw if i["confidence"] > 0.7])
#             img_record.ocr_confidence = float(np.mean([i["confidence"] for i in kw])) if kw else 0.0
#             logger.info(f"✅ OCR [{image_id}]")
#         except Exception as e:
#             logger.warning(f"⚠️  OCR failed [{image_id}]: {e}")
#         try:
#             q = image_quality.assess_overall_quality(file_path)
#             img_record.quality_score = q.get("overall", 0)
#             img_record.quality_level = q.get("quality_level", "Unknown")
#             img_record.sharpness     = q.get("sharpness", 0)
#             img_record.exposure      = q.get("exposure", 0)
#             img_record.contrast      = q.get("contrast", 0)
#             img_record.composition   = q.get("composition", 0)
#             logger.info(f"✅ Quality [{image_id}]: {img_record.quality_level}")
#         except Exception as e:
#             logger.warning(f"⚠️  Quality failed [{image_id}]: {e}")
#         try:
#             # ── GATE: only run emotion model if InsightFace detected real human faces ──
#             # InsightFace (ArcFace) is accurate at human faces — won't detect cat faces,
#             # car headlights, or turtle heads. OpenCV Haar cascade (used by the emotion
#             # engine) will detect ALL of these as "faces" and classify them wrongly.
#             # Using the faces table count ensures we only run on real human images.
#             _face_count_check = db.query(DBFace).filter(DBFace.image_id == image_id).count()
#             if _face_count_check == 0:
#                 # No human faces detected by InsightFace → skip emotion model entirely
#                 img_record.dominant_emotion   = "neutral"
#                 img_record.face_emotion_count = 0
#                 img_record.emotion_data       = "[]"
#                 logger.info(f"⏭️  Emotion [{image_id}]: skipped (no InsightFace faces)")
#             else:
#                 ed = emotion_detection.detect_emotions(file_path)
#                 # ── Confidence filter: only keep high-confidence detections ──────────
#                 # Low-confidence results (< 0.50) from Haar cascade on non-frontal or
#                 # partially visible faces produce random emotion labels. Filter them out.
#                 ed = [e for e in ed if e.get("confidence", 0) >= 0.50]
#                 img_record.face_emotion_count = len(ed)
#                 img_record.dominant_emotion   = ed[0]["emotion"] if ed else "neutral"
#                 img_record.emotion_data       = _json.dumps(ed)
#                 logger.info(f"✅ Emotion [{image_id}]: {img_record.dominant_emotion} ({len(ed)} faces)")
#         except Exception as e:
#             logger.warning(f"⚠️  Emotion failed [{image_id}]: {e}")
#         try:
#             a = aesthetic_scoring.score_aesthetics(file_path)
#             img_record.aesthetic_score  = a.get("aesthetic_score", 0)
#             img_record.aesthetic_rating = a.get("rating", "Unknown")
#             logger.info(f"✅ Aesthetic [{image_id}]: {img_record.aesthetic_rating}")
#         except Exception as e:
#             logger.warning(f"⚠️  Aesthetic failed [{image_id}]: {e}")
#         db.commit()
#         logger.info(f"✅ Enrichment done for image {image_id}")
#     except Exception as e:
#         db.rollback()
#         logger.error(f"❌ Enrichment failed [{image_id}]: {e}", exc_info=True)
#     finally:
#         db.close()


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
#             except Exception:
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
#             logger.info("📊 Recluster triggered (background thread)")
#             # ── Run in a real OS thread, NOT in FastAPI BackgroundTasks ──────
#             # FastAPI BackgroundTasks are sequential: recluster would only start
#             # after _enrich_image (BLIP + OCR, ~15-30s per image) finishes.
#             # With 10 images queued that's 150-300s before recluster even starts.
#             # A daemon thread runs immediately, parallel to enrichment.
#             import threading
#             t = threading.Thread(target=recluster, daemon=True)
#             t.start()
#             recluster_last_triggered = now
#             with open(RECLUSTER_COUNTER_PATH, 'w') as f:
#                 f.write('0')
#     except Exception as e:
#         logger.warning(f"Recluster check failed: {e}")


# # ── Lifespan ──────────────────────────────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     init_db()
#     ensure_extra_columns()          # ← runs safely here, after DB is initialised
#     _startup_fix_filenames()
#     if os.path.exists(FAISS_INDEX_PATH):
#         try:
#             search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
#             logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
#         except Exception as e:
#             logger.error(f"Index load failed: {e}")
#             search_engine.index = None
#     try:
#         _db = SessionLocal()
#         try:
#             faces_total    = _db.query(DBFace).filter(DBFace.face_embedding != None).count()
#             faces_assigned = _db.query(DBFace).filter(DBFace.face_embedding != None, DBFace.person_id != None).count()
#             people_total  = _db.query(Person).count()
#             imgs_in_album = _db.query(DBImage).filter(DBImage.album_id != None).count()
#             total_imgs    = _db.query(DBImage).count()
#             needs_face_repair  = faces_total > 0 and (faces_assigned == 0 or people_total == 0)
#             needs_album_repair = total_imgs  > 0 and imgs_in_album == 0
#         finally:
#             _db.close()
#         if needs_face_repair or needs_album_repair:
#             logger.warning(f"⚠️  Stale data — faces={faces_total} assigned={faces_assigned} people={people_total} imgs_in_album={imgs_in_album}/{total_imgs}")
#             logger.info("🔄 Auto-repairing on startup...")
#             recluster()
#             logger.info("✅ Auto-repair done")
#         else:
#             logger.info(f"✅ DB healthy: {faces_assigned}/{faces_total} faces assigned, {people_total} people, {imgs_in_album}/{total_imgs} imgs in albums")
#     except Exception as e:
#         logger.warning(f"Auto-repair check failed (non-fatal): {e}")
#     logger.info("✅ Ready!")
#     yield


# # ── app = FastAPI() MUST be here, before any app.include_router() ─────────────
# app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)

# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.requests import Request as StarletteRequest

# class FixImagePathMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: StarletteRequest, call_next):
#         path = request.scope["path"]
#         if path.startswith("/image/"):
#             suffix = path[len("/image/"):]
#             suffix = suffix.lstrip("/")
#             if suffix.startswith("images/"):
#                 suffix = suffix[len("images/"):]
#             bare = os.path.basename(suffix)
#             if bare:
#                 request.scope["path"] = f"/image/{bare}"
#         return await call_next(request)

# app.add_middleware(FixImagePathMiddleware)

# if not os.path.exists(IMAGE_DIR):
#     os.makedirs(IMAGE_DIR)

# app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# # ── All routers registered AFTER app = FastAPI() ──────────────────────────────
# app.include_router(dl_router)
# app.include_router(voice_router)
# app.include_router(feat_router)    # ← safely here, app already exists


# # ────────────────────────────────────────────────────────────────────────────
# # UTILITY
# # ────────────────────────────────────────────────────────────────────────────
# @app.get("/health")
# def health():
#     return {"status": "ready", "image_index": search_engine.index.ntotal if search_engine.index else 0}

# @app.get("/debug")
# def debug_db():
#     db = SessionLocal()
#     try:
#         total_images         = db.query(DBImage).count()
#         total_faces          = db.query(DBFace).count()
#         faces_with_embedding = db.query(DBFace).filter(DBFace.face_embedding != None).count()
#         faces_with_image_id  = db.query(DBFace).filter(DBFace.image_id != None).count()
#         faces_with_person    = db.query(DBFace).filter(DBFace.person_id != None).count()
#         total_people         = db.query(Person).count()
#         total_albums         = db.query(Album).count()
#         images_in_album      = db.query(DBImage).filter(DBImage.album_id != None).count()
#         people_detail = []
#         for p in db.query(Person).all():
#             faces   = db.query(DBFace).filter(DBFace.person_id == p.id).all()
#             img_ids = [f.image_id for f in faces if f.image_id]
#             people_detail.append({"id": p.id, "name": p.name, "face_count": len(faces), "faces_with_image_id": len(img_ids)})
#         album_detail = []
#         for a in db.query(Album).all():
#             imgs = db.query(DBImage).filter(DBImage.album_id == a.id).count()
#             album_detail.append({"id": a.id, "title": a.title, "image_count": imgs})
#         return {
#             "images": total_images, "faces": total_faces,
#             "faces_with_embedding": faces_with_embedding, "faces_with_image_id": faces_with_image_id,
#             "faces_with_person": faces_with_person, "people": total_people,
#             "albums": total_albums, "images_in_album": images_in_album,
#             "people_detail": people_detail, "album_detail": album_detail,
#             "action_needed": "Run POST /recluster to fix person/album assignments" if faces_with_person == 0 or images_in_album == 0 else "OK",
#         }
#     finally:
#         db.close()

# @app.get("/test-db")
# def test_db():
#     db = SessionLocal()
#     try:
#         count  = db.query(DBImage).count()
#         images = db.query(DBImage).limit(1).all()
#         return {
#             "status": "ok", "total_images": count,
#             "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None,
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # IMAGE LISTING & SERVING
# # ────────────────────────────────────────────────────────────────────────────
# @app.get("/images")
# def get_all_images(limit: int = 100):
#     db = SessionLocal()
#     try:
#         images = _live(db).limit(limit).all()
#         return {"results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "face_emotion_count": img.face_emotion_count, "aesthetic_score": img.aesthetic_score, "aesthetic_rating": img.aesthetic_rating, "caption_short": img.caption_short, "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
#     finally:
#         db.close()

# @app.get("/image/{filename:path}")
# def get_image_file(filename: str):
#     bare = os.path.basename(filename.lstrip("/"))
#     if not bare or bare in (".", ".."):
#         raise HTTPException(status_code=400, detail="Invalid filename")
#     candidates = [
#         os.path.join(IMAGE_DIR, bare),
#         os.path.normpath(os.path.join(os.getcwd(), "..", "data", "images", bare)),
#         os.path.normpath(os.path.join(os.getcwd(), "data", "images", bare)),
#         os.path.normpath(os.path.join(os.getcwd(), "images", bare)),
#     ]
#     for p in candidates:
#         if os.path.exists(p):
#             return FileResponse(p)
#     db = SessionLocal()
#     try:
#         img = db.query(DBImage).filter((DBImage.filename == bare) | (DBImage.filename == f"/images/{bare}")).first()
#         if img and img.original_path:
#             p = os.path.normpath(img.original_path.replace("//", "/"))
#             if os.path.exists(p):
#                 return FileResponse(p)
#             alt = os.path.join(IMAGE_DIR, os.path.basename(img.original_path))
#             if os.path.exists(alt):
#                 return FileResponse(alt)
#     finally:
#         db.close()
#     logger.warning(f"404 image bare={bare!r} IMAGE_DIR={IMAGE_DIR}")
#     raise HTTPException(status_code=404, detail=f"Image not found: {bare}")

# @app.get("/timeline")
# def get_timeline():
#     db = SessionLocal()
#     try:
#         images = _live(db).order_by(DBImage.timestamp.desc()).all()
#         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "face_emotion_count": img.face_emotion_count, "aesthetic_score": img.aesthetic_score, "aesthetic_rating": img.aesthetic_rating, "caption_short": img.caption_short, "ocr_text_enhanced": img.ocr_text_enhanced, "person_count": img.person_count, "width": img.width, "height": img.height, "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # DELETE / TRASH / RESTORE
# # ────────────────────────────────────────────────────────────────────────────
# @app.post("/delete_image")
# def delete_image(image_id: int = Form(...)):
#     db = SessionLocal()
#     try:
#         img = db.query(DBImage).filter(DBImage.id == image_id).first()
#         if not img:
#             raise HTTPException(status_code=404, detail="Image not found")
#         img.is_trashed = True
#         img.trashed_at = datetime_module.datetime.now()
#         db.commit()
#         return {"success": True, "message": "Image moved to trash"}
#     finally:
#         db.close()

# @app.post("/restore")
# def restore_image(image_id: int = Form(...)):
#     db = SessionLocal()
#     try:
#         img = db.query(DBImage).filter(DBImage.id == image_id).first()
#         if not img:
#             raise HTTPException(status_code=404, detail="Image not found")
#         img.is_trashed = False
#         img.trashed_at = None
#         db.commit()
#         return {"success": True, "message": "Image restored"}
#     finally:
#         db.close()

# @app.post("/permanent_delete")
# def permanent_delete(image_id: int = Form(...)):
#     db = SessionLocal()
#     try:
#         img = db.query(DBImage).filter(DBImage.id == image_id).first()
#         if not img:
#             raise HTTPException(status_code=404, detail="Image not found")
#         filename  = img.filename
#         file_path = img.original_path or os.path.join(IMAGE_DIR, filename)
#         bare = os.path.basename(file_path)
#         for p in [file_path, os.path.join(IMAGE_DIR, bare)]:
#             if p and os.path.exists(p):
#                 try:
#                     os.remove(p)
#                     break
#                 except Exception as e:
#                     logger.warning(f"Could not delete file: {e}")
#         if search_engine.index is not None:
#             try:
#                 # Fast: remove just this one vector — no need to re-embed everything
#                 ids_arr = np.array([image_id], dtype="int64")
#                 search_engine.index.remove_ids(ids_arr)
#                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
#                 logger.info(f"✅ FAISS vector {image_id} removed")
#             except Exception as e:
#                 logger.warning(f"FAISS remove (non-critical): {e}")
#         db.query(DBFace).filter(DBFace.image_id == image_id).delete()
#         db.delete(img)
#         db.commit()
#         return {"success": True, "message": f"Image '{filename}' permanently deleted"}
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         db.close()

# @app.get("/trash")
# def get_trash():
#     db = SessionLocal()
#     try:
#         images = db.query(DBImage).filter(DBImage.is_trashed == True).all()
#         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "trashed_at": img.trashed_at.isoformat() if img.trashed_at else None, "timestamp": img.timestamp.isoformat() if img.timestamp else None, "caption": img.caption_short or ""} for img in images]}
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # FAVORITES
# # ────────────────────────────────────────────────────────────────────────────
# @app.post("/toggle_favorite")
# def toggle_favorite(image_id: int = Form(...)):
#     db = SessionLocal()
#     try:
#         img = db.query(DBImage).filter(DBImage.id == image_id).first()
#         if not img:
#             raise HTTPException(status_code=404, detail="Image not found")
#         img.is_favorite = not img.is_favorite
#         db.commit()
#         return {"success": True, "is_favorite": img.is_favorite}
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
#         images = _live(db).filter(DBImage.is_favorite == True).all()
#         return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # SEARCH
# # ────────────────────────────────────────────────────────────────────────────
# def _score_candidates(query_emb, sig_words, query_colors, db, top_k, extra_emb=None, text_weight=1.0, image_weight=0.0):
#     if search_engine.index is None:
#         return []
#     if extra_emb is not None:
#         blended = text_weight * query_emb + image_weight * extra_emb
#         norm = np.linalg.norm(blended)
#         if norm > 1e-8:
#             blended /= norm
#         q_vec = blended
#     else:
#         q_vec = query_emb
#     total = search_engine.index.ntotal
#     q = q_vec.reshape(1, -1).astype("float32")
#     faiss.normalize_L2(q)
#     distances, indices = search_engine.index.search(q, total)
#     all_candidates = []
#     for dist, idx in zip(distances[0], indices[0]):
#         if idx == -1:
#             continue
#         clip_score = float(dist)
#         if clip_score < CLIP_SCORE_MIN:
#             break
#         img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
#         if not img or img.is_trashed:
#             continue
#         # (empty-caption guard removed — was killing valid results for colour queries)
#         # ── OCR bonus (only if most sig_words appear) ──────────────────────
#         ocr_bonus = 0.0
#         ocr_src = (img.ocr_text_enhanced or "") + " " + (getattr(img, "ocr_keywords", "") or "")
#         if sig_words and ocr_src.strip():
#             txt  = ocr_src.lower()
#             hits = sum(1 for w in sig_words if w in txt)
#             # Require at least half the words to match to avoid false boosts
#             if hits >= max(1, len(sig_words) * 0.5):
#                 ocr_bonus = min(hits / len(sig_words), 1.0)

#         # ── Tag bonus ──────────────────────────────────────────────────────
#         tag_bonus = 0.0
#         all_tags_text = " ".join(_user_tags(img))
#         if img.scene_label:
#             all_tags_text += " " + img.scene_label.lower()
#         if all_tags_text.strip() and any(w in all_tags_text for w in sig_words):
#             tag_bonus = 1.0

#         # ── Caption bonus + contradiction filter ─────────────────────────
#         caption_bonus = 0.0
#         hard_exclude    = False  # True = skip this result entirely
#         caption_bonus   = 0.0
#         caption_penalty = 0.0
#         caption_src = " ".join(filter(None, [
#             img.caption_short or "", img.caption_detailed or ""
#         ])).lower()

#         # ── Shared sets (always defined — needed even when sig_words=[]) ──
#         def _stem(w):
#             if len(w) > 4 and w.endswith("ses"): return w[:-2]
#             if len(w) > 3 and w.endswith("ies"): return w[:-3]+"y"
#             if len(w) > 3 and w.endswith("es"):  return w[:-2]
#             if len(w) > 3 and w.endswith("s"):   return w[:-1]
#             return w

#         _COLOURS = {"black","white","red","blue","green","yellow","orange",
#                     "purple","pink","brown","gray","grey","golden","silver",
#                     "nude","beige","cream","navy","teal","maroon","violet"}
#         _ANIMALS = {
#             "horse","cow","dog","cat","bird","elephant","tiger","lion",
#             "bear","wolf","deer","sheep","goat","pig","rabbit","snake",
#             "fish","chicken","duck","frog","butterfly","spider",
#             "fox","monkey","giraffe","zebra","panda","kangaroo","crocodile",
#             "alligator","rhino","hippo","whale","dolphin","shark","eagle",
#             "parrot","penguin","owl","bat","beetle","ant","bee","crab",
#             "donkey","camel","llama","raccoon","squirrel","hamster",
#             "mouse","rat","gecko","lizard","tortoise","turtle","swan",
#             "otter","seal","walrus","beaver","meerkat","flamingo","peacock",
#             "cheetah","leopard","jaguar","gorilla","chimpanzee","baboon",
#             "koala","wombat","hedgehog","ferret","chinchilla","iguana",
#             "chameleon","scorpion","jellyfish","octopus","lobster","shrimp",
#             "kitten","puppy","calf","foal","piglet","lamb","cub","chick",
#         }
#         _CLOTHING = {"dress","suit","shirt","jacket","sari","skirt","gown",
#                      "uniform","coat","blouse","top","hoodie","sweater",
#                      "saree","kurta","lehenga","tuxedo","blazer","vest",
#                      "bikini","swimsuit","robe","kimono","pyjama","shorts",
#                      "trouser","pant","jeans","legging"}
#         _GENDER = {"woman","girl","lady","female","man","boy","male","guy"}
#         _FEMALE = {"woman","girl","lady","female"}
#         _MALE   = {"man","boy","guy","male"}
#         _HUMAN_WORDS = {"man","woman","boy","girl","person","people","lady",
#                         "guy","child","adult","player","actor","actress"}

#         # Always compute q_* from sig_words (empty set when sig_words=[])
#         _q_set     = set(sig_words)
#         q_colours  = _COLOURS  & _q_set
#         q_animals  = _ANIMALS  & _q_set
#         q_clothing = _CLOTHING & _q_set
#         q_gender   = _GENDER   & _q_set

#         # Always compute cap_words from caption
#         if caption_src:
#             _cap_raw  = caption_src.split()
#             cap_words = {_stem(w) for w in _cap_raw} | set(_cap_raw)
#         else:
#             cap_words = set()

#         if sig_words and caption_src:
#             # ── Caption bonus ────────────────────────────────────────────
#             hits = sum(1 for w in _q_set if w in caption_src)
#             match_ratio = hits / len(_q_set)
#             if match_ratio >= 0.5:
#                 caption_bonus = match_ratio

#             # ── 1. Animal mismatch ───────────────────────────────────────
#             if q_animals:
#                 cap_animals = _ANIMALS & cap_words
#                 if cap_animals and not (cap_animals & q_animals):
#                     hard_exclude = True

#             # ── 2. Colour + clothing contradiction (context-aware) ──────
#             if not hard_exclude and q_colours:
#                 cap_colours  = _COLOURS & cap_words
#                 cap_clothing = _CLOTHING & cap_words
#                 same_cloth   = bool(cap_clothing & q_clothing)
#                 contradicting = cap_colours - q_colours
#                 matching      = cap_colours & q_colours
#                 if same_cloth and cap_colours:
#                     # Caption has same clothing item → colour must match
#                     # e.g. "white dress" for "black dress" → EXCLUDED
#                     # "black hair wearing white dress" → "black" matches query but 
#                     # the DRESS colour (white) contradicts → still exclude
#                     if contradicting and not matching:
#                         hard_exclude = True
#                     elif contradicting and matching:
#                         caption_penalty += 0.10
#                 elif not same_cloth and cap_colours and contradicting and not matching:
#                     caption_penalty += 0.18

#                         # ── 3. Clothing constraint — hard exclude on mismatch ────────
#             if not hard_exclude and q_clothing:
#                 cap_clothing = _CLOTHING & cap_words
#                 if cap_clothing and not (cap_clothing & q_clothing):
#                     # Caption names a DIFFERENT clothing item (e.g. "suit" for "dress")
#                     hard_exclude = True
#                 elif not cap_clothing:
#                     # Caption has NO clothing word at all
#                     if q_colours and (q_colours & cap_words):
#                         # Colour word appears in caption (e.g. "red eyes") but no clothing
#                         # → the colour belongs to something else, not the dress → exclude
#                         hard_exclude = True
#                     elif q_colours and not (q_colours & cap_words):
#                         # No clothing AND wrong/no colour → definitely not the right image
#                         hard_exclude = True

#             # ── 4. Gender mismatch ───────────────────────────────────────
#             if not hard_exclude and q_gender:
#                 q_female = bool(_FEMALE & q_gender)
#                 q_male   = bool(_MALE   & q_gender)
#                 if q_female and (_MALE & cap_words) and not (_FEMALE & cap_words):
#                     hard_exclude = True   # query=woman but caption has only men → always exclude
#                 if q_male and (_FEMALE & cap_words) and not (_MALE & cap_words):
#                     hard_exclude = True   # query=man but caption has only women → always exclude

#             if caption_penalty >= 0.40:
#                 hard_exclude = True
#             caption_penalty = min(caption_penalty, 0.38)

#         # ── Animal query: exclude human-only captions (always runs) ──────
#         if not hard_exclude and q_animals and not q_gender and caption_src:
#             cap_has_human  = bool(_HUMAN_WORDS & cap_words)
#             cap_has_animal = bool(_ANIMALS & cap_words)
#             if cap_has_human and not cap_has_animal:
#                 hard_exclude = True

#         # ── BUGFIX: Human query: exclude animal-only captions ───────────
#         if not hard_exclude and (_HUMAN_WORDS & _q_set) and caption_src:
#             cap_has_human  = bool(_HUMAN_WORDS & cap_words)
#             cap_has_animal = bool(_ANIMALS & cap_words)
#             if cap_has_animal and not cap_has_human:
#                 hard_exclude = True

#         # ── BUGFIX: Empty/missing caption — use scene_label as fallback ─
#         # When caption_src is empty (BLIP not yet run or failed), ALL the
#         # animal/human filters above are skipped because they check
#         # `if sig_words and caption_src`.  Non-matching animals then leak
#         # through with a pure CLIP score.  Use scene_label + user_tags to
#         # catch the worst offenders.
#         if not hard_exclude and not caption_src and q_animals:
#             # Build a stem-aware token set from scene_label + user_tags
#             _sl = (img.scene_label or "").lower()
#             _ut = " ".join(_user_tags(img)).lower()
#             _fb_raw = set((_sl + " " + _ut).split())
#             _fb_toks = _fb_raw | {_stem(w) for w in _fb_raw}
#             _fb_animals = _ANIMALS & _fb_toks
#             # If scene_label names a DIFFERENT animal from the query → exclude
#             if _fb_animals and not (_fb_animals & q_animals):
#                 hard_exclude = True
#             # If scene_label has NO animal at all but has human words → exclude
#             # (e.g. "person, sink" scene_label for a cat query)
#             if not _fb_animals:
#                 _fb_humans = _HUMAN_WORDS & _fb_toks
#                 if _fb_humans:
#                     hard_exclude = True

#         # ── BUGFIX: Specific object words absent from caption ────────────────
#         # If query mentions a concrete specific object (flower, rose, guitar, etc.)
#         # and the caption exists but doesn't mention it at all → CLIP is hallucinating
#         # semantic similarity. These concrete words almost never produce false negatives
#         # (if the image has a flower, BLIP will say "flower").
#         _SPECIFIC_OBJECTS = {
#             "flower","rose","tulip","sunflower","daisy","bouquet","petal",
#             "guitar","piano","violin","drum","trumpet","saxophone",
#             "cake","pizza","burger","sandwich","coffee","wine","beer","cocktail",
#             "trophy","medal","cup","award","crown",
#             "balloon","umbrella","candle","lantern","flag",
#             "sword","shield","arrow","bow",
#             "basketball","football","soccer","tennis","cricket","baseball",
#             "bicycle","motorcycle","skateboard","surfboard","snowboard",
#         }
#         if not hard_exclude and caption_src and sig_words:
#             _q_specific = _SPECIFIC_OBJECTS & set(sig_words)
#             if _q_specific:
#                 # Any of the specific objects must appear in caption (stemmed)
#                 _q_specific_stemmed = {_stem(w) for w in _q_specific} | _q_specific
#                 if not (_q_specific_stemmed & cap_words):
#                     hard_exclude = True

#         # ── Hard exclude — skip entirely ────────────────────────────────
#         if hard_exclude:
#             continue

#         # ── Color bonus ──────────────────────────────────────────────────
#         color_bonus = 0.0
#         if query_colors and getattr(img, "avg_r", None) is not None:
#             img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
#             for qc in query_colors:
#                 d = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
#                 color_bonus = max(color_bonus, max(0.0, 1.0 - d / np.sqrt(3)))

#         # ── Final score ──────────────────────────────────────────────────
#         exact_match_bonus = 0.0
#         if sig_words and caption_src:
#             all_present = all(w in caption_src for w in sig_words)
#             majority_present = sum(1 for w in sig_words if w in caption_src) / len(sig_words) >= 0.7
#             if all_present:
#                 exact_match_bonus = 0.25
#             elif majority_present:
#                 exact_match_bonus = 0.12

#         # Dynamic colour weight: when query explicitly asks for a colour
#         # (e.g. "white horse", "red car"), pixel colour scoring should matter
#         # much more than the default 0.01. Raise it to 0.15 so a brown horse
#         # scores significantly lower than a white horse for "white horse".
#         _colour_weight = 0.15 if query_colors else 0.01

#         final_score = (0.60 * clip_score
#                       + 0.20 * caption_bonus
#                       + 0.15 * exact_match_bonus
#                       + 0.03 * tag_bonus
#                       + 0.01 * ocr_bonus
#                       + _colour_weight * color_bonus
#                       - caption_penalty)
#         all_candidates.append({"img": img, "clip": clip_score, "final": final_score})
#     all_candidates.sort(key=lambda x: x["final"], reverse=True)
#     kept = [c for c in all_candidates if c["final"] >= FINAL_SCORE_MIN]
#     if all_candidates:
#         logger.info(f"📊 CLIP: {len(all_candidates)} above CLIP_MIN, {len(kept)} above FINAL_MIN={FINAL_SCORE_MIN}, top={all_candidates[0]['final']:.3f} bottom={all_candidates[-1]['final']:.3f}")
#     results = []
#     for c in kept[:top_k]:
#         img = c["img"]
#         results.append({
#                 "id": img.id,
#                 "filename": _img_url(img.filename),
#                 "score": round(c["final"] * 100, 2),
#                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
#                 "person_count": img.person_count or 0,
#                 "caption_short": img.caption_short or "",
#                 "caption_detailed": img.caption_detailed or "",
#                 "quality_score": img.quality_score or 0,
#                 "quality_level": img.quality_level or "",
#                 "sharpness": img.sharpness or 0,
#                 "exposure": img.exposure or 0,
#                 "contrast": img.contrast or 0,
#                 "composition": img.composition or 0,
#                 "dominant_emotion": img.dominant_emotion or "",
#                 "face_emotion_count": img.face_emotion_count or 0,
#                 "aesthetic_score": img.aesthetic_score or 0,
#                 "aesthetic_rating": img.aesthetic_rating or "",
#                 "ocr_text_enhanced": img.ocr_text_enhanced or "",
#                 "scene_label": img.scene_label or "",
#                 "width": img.width,
#                 "height": img.height,
#                 "is_favorite": bool(img.is_favorite),
#                 "user_tags": _user_tags(img),
#                 "photo_note": getattr(img, "photo_note", "") or "",
#             })
#     return results

# @app.post("/search")
# def search(query: str = Form(...), top_k: int = Form(20)):
#     if not query or not query.strip():
#         return {"status": "error", "message": "Query empty"}

#     # ── Emotion search: emoji OR text keywords → filter by dominant_emotion ──
#     detected_emotion = _extract_emotion_from_query(query)
#     is_pure_emoji    = query.strip() in EMOTION_EMOJI_MAP

#     # Check if query is MOSTLY about emotion (emotion word + optional "faces"/"photos" etc)
#     EMOTION_MODIFIERS = {"faces", "face", "photos", "photo", "pictures", "picture",
#                          "images", "image", "people", "person", "moments", "looking",
#                          "expression", "expressions", "ones", "me", "us", "all",
#                          # BUGFIX: common query prefixes that were left in non_emotion_words
#                          # causing "show me happy faces" / "find sad photos" to miss the
#                          # emotion route and fall through to CLIP which returns nothing.
#                          "show", "find", "get", "give", "display", "share",
#                          "my", "the", "a", "with", "some", "any"}
#     query_words = set(query.lower().split())
#     non_emotion_words = query_words - EMOTION_MODIFIERS -         {kw for kws in EMOTION_KEYWORDS.values() for kw in kws}
#     is_emotion_query = detected_emotion and len(non_emotion_words) == 0

#     if detected_emotion and (is_pure_emoji or is_emotion_query):
#         # Route directly to emotion DB filter
#         # Only match images where emotion was detected by face analysis
#         # (face_emotion_count > 0 means actual faces were detected and analysed)
#         db = SessionLocal()
#         try:
#             # Require: actual faces detected (face_emotion_count>0) AND human present (person_count>0)
#             # This prevents turtles/cars/cats from appearing in happy/angry emotion search
#             # Strict: only images where faces were actually detected + analysed
#             # face_emotion_count > 0 = real face detection ran on this image
#             imgs = _live(db).filter(
#                 DBImage.dominant_emotion == detected_emotion,
#                 DBImage.face_emotion_count > 0,
#             ).order_by(DBImage.timestamp.desc()).limit(top_k).all()
#             # NO fallback that removes face_emotion_count requirement.
#             # Without this guard, cars/turtles/spider-man with wrongly assigned
#             # emotion labels appear in results. Better to show fewer real results.
#             if imgs:
#                 return {"status": "found", "query": query, "count": len(imgs),
#                         "results": [{"id": img.id, "filename": _img_url(img.filename),
#                                      "score": 90.0,
#                                      "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                                      "caption_short": img.caption_short or "",
#                                      "user_tags": _user_tags(img),
#                                      "dominant_emotion": img.dominant_emotion,
#                                      "person_count": img.person_count or 0,
#                                      "photo_note": getattr(img,"photo_note","") or ""} for img in imgs]}
#             else:
#                 # No results — emotions may not be detected yet
#                 return {"status": "not_found",
#                         "message": f"No photos with '{detected_emotion}' emotion found. "
#                                    f"Click 'Fix Emotions' in the sidebar to detect emotions on your photos."}
#         finally:
#             db.close()

#     # ── Expand scene emojis to text for CLIP ────────────────────────────────
#     expanded_query = _expand_query_emojis(query)
#     # Expand sport/activity synonyms for better CLIP recall
#     _words = expanded_query.lower().split()
#     if any(w in SPORT_SYNONYMS for w in _words):
#         _words_exp = []
#         for w in _words:
#             _words_exp.append(SPORT_SYNONYMS.get(w, w))
#         expanded_query = " ".join(_words_exp)
#     processed_query = _clean_query(expanded_query)
#     logger.info(f"🔍 Search: '{query}' → '{processed_query}'")

#     # BUGFIX: q_tokens_pre must be defined HERE — it is used by the
#     # should_search_people pre-computation block below, which runs before
#     # the later assignment at the CLIP section (line ~17609).
#     q_tokens_pre = query.lower().strip().split()

#     # If emotion emoji mixed with other words, boost emotion filter
#     emotion_boost = detected_emotion  # may be None

#     # ── Pre-compute should_search_people so sig_words can use it ────────────
#     # (Full person-search logic runs later; here we just need the flag)
#     _DESCRIPTOR_PRE = {
#         "a","an","the","in","on","at","of","for","with","by","to","from","into",
#         "is","are","was","were","be","been","has","have","had","do","does","did",
#         "will","would","could","should","may","might","can","wearing","holding",
#         "standing","sitting","posing","looking","long","short","tall","small",
#         "big","large","little","young","old","beautiful","pretty","black","white",
#         "red","blue","green","yellow","orange","purple","pink","brown","gray",
#         "woman","man","girl","boy","lady","guy","person","people","female","male",
#         "human","child","adult","baby","player","actor","photo","image","picture",
#         "standing","next","front","back","window","field","camera","dress","suit",
#         "shirt","jacket","hair","smiling",
#         # FIX: query-prefix words that aren't person names
#         "show","find","get","give","display","search","look","me","my","us","our",
#         # FIX: emotion/state adjectives that aren't person names
#         "happy","sad","angry","surprised","fearful","neutral","disgusted",
#         "smiling","laughing","crying","excited","bored","tired","scared",
#         # FIX: common descriptive adjectives that confused the name detector
#         "funny","cute","cool","nice","great","good","bad","best","worst",
#         "images","photos","pictures","moments","all","some","any","many",
#     }
#     # Animal/object words that look like names but are NOT people
#     _NON_PERSON_WORDS = {
#         # animals
#         "cat","dog","horse","cow","bird","fish","fox","lion","tiger","bear","wolf",
#         "pig","rabbit","duck","frog","snake","deer","sheep","goat","chicken","otter",
#         "seal","panda","koala","monkey","giraffe","zebra","elephant","whale","shark",
#         "eagle","owl","penguin","parrot","bee","ant","spider","butterfly","kitten",
#         "puppy",
#         # nature / places
#         "sunset","beach","ocean","mountain","forest","river","flower","tree",
#         "sky","snow","rain","night","city","park","road","bridge","building","house",
#         # vehicles / objects
#         "car","bus","train","plane","boat","food","pizza","cake","coffee",
#         # fictional characters & universes — prevent person-search triggering
#         "ironman","spiderman","batman","superman","thor","hulk","avengers",
#         "marvel","anime","cartoon","fictional","hero","villain","superhero",
#         "comic","spider","thanos","loki","deadpool","wolverine","aquaman",
#         "flash","wonder","hawkeye","ultron","venom","groot","rocket","nebula",
#         "gamora","starlord","antman","blackwidow","scarlet","vision","warmachine",
#         "falcon","panther","shuri","okoye","wakanda","asgard","xmen","cyclops",
#         "magneto","mystique","beast","nightcrawler","storm","phoenix","rogue",
#         # music genres / scene words (not person names)
#         "kpop","kdrama","jpop","bollywood","hollywood","netflix","disney","pixar",
#         "hiphop","hiphop","jazz","rock","pop","metal","indie","classical",
#         # common OCR / UI words that appear as image text
#         "user","login","password","email","search","home","menu","settings",
#         "button","click","submit","cancel","back","next","done","ok","yes","no",
#         "error","loading","please","enter","select","upload","download","share",
#         "follow","like","comment","post","profile","account","app","website",
#         # generic concept words
#         "vintage","retro","modern","classic","dark","light","minimal","abstract",
#         "nature","urban","street","indoor","outdoor","close","wide","macro",
#     }
#     _min_len_pre      = 3 if len(q_tokens_pre) <= 3 else 4
#     _nc_pre           = [w for w in q_tokens_pre if len(w) >= _min_len_pre
#                          and w not in _DESCRIPTOR_PRE
#                          and w not in _NON_PERSON_WORDS
#                          and w.isalpha()]
#     _dc_pre           = sum(1 for w in q_tokens_pre if w in _DESCRIPTOR_PRE)
#     _is_desc_pre      = _dc_pre >= len(q_tokens_pre) * 0.5
#     should_search_people = bool(_nc_pre) and not _is_desc_pre


#     # ── Tag fast-path ─────────────────────────────────────────────────────────
#     # BUGFIX: Arbitrary user tags (e.g. "birthday", "vacation", "grandma") only
#     # contributed a 0.03 weight bonus to CLIP scores, making them essentially
#     # invisible when typed in the search bar. Now: if ANY word in the query
#     # exactly matches a known user tag in the DB, we return those tagged images
#     # immediately (score=97) merged with any further CLIP/OD results.
#     # This runs BEFORE the OD and CLIP paths so tags always take priority.
#     _q_words_lower   = set(query.lower().strip().split())
#     _tag_fast_results = []
#     _tag_fast_ids     = set()
#     try:
#         _db_tag = SessionLocal()
#         try:
#             # Collect all distinct tags that exist in the DB
#             _all_tag_rows = _db_tag.execute(text(
#                 "SELECT DISTINCT user_tags FROM images "
#                 "WHERE user_tags IS NOT NULL AND user_tags != '[]' "
#                 "AND (is_trashed IS NULL OR is_trashed=0)"
#             )).fetchall()
#             import json as _json_tag
#             _known_tags = set()
#             for _trow in _all_tag_rows:
#                 try:
#                     for _t in _json_tag.loads(_trow[0] or "[]"):
#                         _known_tags.add(_t.strip().lower())
#                 except Exception:
#                     pass
#             # Find query words that are known tags
#             _matched_tags = _q_words_lower & _known_tags
#             if _matched_tags:
#                 for _img in _live(_db_tag).all():
#                     _itags = {t.strip().lower() for t in _user_tags(_img)}
#                     if _itags & _matched_tags:
#                         if _img.id not in _tag_fast_ids:
#                             _tag_fast_ids.add(_img.id)
#                             _tag_fast_results.append({
#                                 "id": _img.id,
#                                 "filename": _img_url(_img.filename) or "",
#                                 "score": 97.0,
#                                 "timestamp": _img.timestamp.isoformat() if _img.timestamp else None,
#                                 "caption_short": _img.caption_short or "",
#                                 "person_count": _img.person_count or 0,
#                                 "dominant_emotion": _img.dominant_emotion or "",
#                                 "quality_level": _img.quality_level or "",
#                                 "quality_score": _img.quality_score or 0,
#                                 "aesthetic_score": _img.aesthetic_score or 0,
#                                 "user_tags": _user_tags(_img),
#                                 "photo_note": getattr(_img, "photo_note", "") or "",
#                                 "is_favorite": bool(_img.is_favorite),
#                                 "scene_label": _img.scene_label or "",
#                                 "width": _img.width, "height": _img.height,
#                             })
#                 logger.info(f"🏷️ Tag fast-path: matched tags {_matched_tags} → {len(_tag_fast_results)} images")
#         finally:
#             _db_tag.close()
#     except Exception as _tag_err:
#         logger.warning(f"Tag fast-path failed (non-fatal): {_tag_err}")

#     # ── Object-detection label fast-path ──────────────────────────────────
#     # Check if any query word directly maps to a COCO-detected object.
#     # scene_label contains comma-separated Faster R-CNN detections (e.g. "cat, person").
#     # This is far more reliable than CLIP for specific animals/objects.
#     QUERY_TO_COCO = {
#         "cat":"cat","cats":"cat","kitten":"cat","kitty":"cat","kittens":"cat",
#         "dog":"dog","dogs":"dog","puppy":"dog","puppies":"dog","pup":"dog",
#         "horse":"horse","horses":"horse","pony":"horse",
#         "bird":"bird","birds":"bird","parrot":"bird","eagle":"bird","owl":"bird",
#         "cow":"cow","cows":"cow","bull":"cow","cattle":"cow",
#         "elephant":"elephant","bear":"bear","zebra":"zebra","giraffe":"giraffe",
#         "sheep":"sheep","goat":"sheep","deer":"deer","rabbit":"rabbit",
#         "soccer":"sports ball","football":"sports ball","sport":"sports ball",
#         "cricket":"baseball bat","baseball":"baseball bat",
#         "tennis":"tennis racket","racket":"tennis racket",
#         "surfing":"surfboard","surf":"surfboard","skateboard":"skateboard",
#         "skiing":"skis","ski":"skis","snowboard":"snowboard",
#         "car":"car","cars":"car","truck":"truck","bus":"bus",
#         "motorcycle":"motorcycle","bicycle":"bicycle","bike":"bicycle",
#         "boat":"boat","ship":"boat","train":"train",
#         "airplane":"airplane","plane":"airplane","aircraft":"airplane",
#         "pizza":"pizza","cake":"cake","sandwich":"sandwich",
#         "apple":"apple","banana":"banana","orange":"orange",
#         "bottle":"bottle","cup":"cup","wine":"wine glass",
#         "laptop":"laptop","keyboard":"keyboard","mouse":"mouse",
#         "phone":"cell phone","cellphone":"cell phone",
#         "chair":"chair","sofa":"couch","couch":"couch","bed":"bed",
#         "tv":"tv","television":"tv","monitor":"tv",
#         "book":"book","books":"book","clock":"clock","vase":"vase",
#         "scissors":"scissors","knife":"knife","fork":"fork","spoon":"spoon",
#     }
#     _q_lower_words = set(query.lower().split())
#     _coco_targets = set()
#     for _w in _q_lower_words:
#         if _w in QUERY_TO_COCO:
#             _coco_targets.add(QUERY_TO_COCO[_w])
    
#     if _coco_targets and not should_search_people:
#         # Direct object detection search — bypass CLIP entirely for these queries
#         _db_od = SessionLocal()
#         try:
#             _od_results = []
#             _seen_od    = set()   # dedup by image id
#             _seen_fn    = set()   # dedup by filename (catches re-indexed duplicates)

#             # Broad animal vocabulary used by BLIP captions — used below to
#             # verify that the COCO model's scene_label isn't a misclassification.
#             _BROAD_ANIMALS = {
#                 "horse","cow","dog","cat","bird","elephant","tiger","lion","bear",
#                 "wolf","deer","sheep","goat","pig","rabbit","snake","fish","chicken",
#                 "duck","frog","fox","monkey","giraffe","zebra","panda","otter",
#                 "squirrel","raccoon","hamster","mouse","rat","kitten","puppy","calf",
#                 "foal","piglet","lamb","cub","chick","parrot","penguin","owl","bat",
#                 "seal","beaver","meerkat","flamingo","peacock","cheetah","leopard",
#                 "jaguar","gorilla","koala","hedgehog","iguana","chameleon","scorpion",
#                 "jellyfish","octopus","lobster","crab","shrimp","bison","buffalo",
#                 "moose","elk","reindeer","caribou","hyena","wildebeest","gazelle",
#             }

#             def _od_row(img, score):
#                 return {
#                     "id": img.id,
#                     "filename": _img_url(img.filename),
#                     "score": score,
#                     "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                     "caption_short": img.caption_short or "",
#                     "person_count": img.person_count or 0,
#                     "dominant_emotion": img.dominant_emotion or "",
#                     "quality_level": img.quality_level or "",
#                     "quality_score": img.quality_score or 0,
#                     "aesthetic_score": img.aesthetic_score or 0,
#                     "user_tags": _user_tags(img),
#                     "photo_note": getattr(img, "photo_note", "") or "",
#                     "is_favorite": bool(img.is_favorite),
#                 }

#             for _img in _live(_db_od).all():
#                 fn = _img_url(_img.filename) or ""
#                 # ── BUGFIX: dedup by filename to catch re-indexed duplicates ──
#                 if _img.id in _seen_od or fn in _seen_fn:
#                     continue

#                 # ── BUGFIX: also match via user_tags (manual tags bypass OD) ──
#                 _img_tags = {t.strip().lower() for t in _user_tags(_img)}
#                 is_tag_match = bool(_img_tags & _coco_targets)

#                 # ── scene_label match (original OD path) ─────────────────────
#                 is_od_match = False
#                 _match_score = 0
#                 if _img.scene_label:
#                     _detected = {x.strip().lower() for x in _img.scene_label.split(",")}
#                     _match_score = sum(1 for t in _coco_targets if t in _detected)
#                     is_od_match = _match_score > 0

#                 if not is_od_match and not is_tag_match:
#                     continue

#                 # ── BUGFIX: caption-verify OD matches to filter COCO misclassifications ──
#                 # Faster R-CNN doesn't know fox/otter/squirrel/rabbit/panda (not COCO
#                 # classes). It often mislabels them as "cat" or "dog". BLIP captions
#                 # are far more accurate — if the caption names a DIFFERENT animal,
#                 # the scene_label detection is a false positive. Skip it.
#                 # Exception: user-tagged images are always trusted.
#                 if is_od_match and not is_tag_match:
#                     _cap = (_img.caption_short or "").lower()
#                     if _cap:
#                         _cap_toks_raw = set(_cap.split())
#                         def _stem_tok(w):
#                             if len(w) > 4 and w.endswith("ses"): return w[:-2]
#                             if len(w) > 4 and w.endswith("ies"): return w[:-3]+"y"
#                             if len(w) > 3 and w.endswith("es"):  return w[:-2]
#                             if len(w) > 3 and w.endswith("s"):   return w[:-1]
#                             return w
#                         _cap_toks = _cap_toks_raw | {_stem_tok(w) for w in _cap_toks_raw}
#                         _cap_animals = _BROAD_ANIMALS & _cap_toks
#                         if _cap_animals and not (_cap_animals & _coco_targets):
#                             logger.info(
#                                 f"🚫 OD-filter: id={_img.id} caption has {_cap_animals}, "
#                                 f"not {_coco_targets} — skipping misclassification"
#                             )
#                             continue
#                     else:
#                         # Caption empty/missing — fall back to scene_label cross-check.
#                         # If scene_label contains ONLY human labels and no animal,
#                         # the "cat"/"dog" detection is almost certainly a false positive.
#                         _sl_raw = (_img.scene_label or "").lower()
#                         _sl_toks = set(_sl_raw.replace(",", " ").split())
#                         _sl_animals = _BROAD_ANIMALS & _sl_toks
#                         _HUMAN_SL   = {"person", "man", "woman", "boy", "girl", "people"}
#                         _sl_humans  = _HUMAN_SL & _sl_toks
#                         if _sl_humans and not _sl_animals:
#                             logger.info(
#                                 f"🚫 OD-filter (no caption): id={_img.id} scene_label has "
#                                 f"only humans {_sl_humans}, not {_coco_targets} — skipping"
#                             )
#                             continue

#                 # FIX Bug2: caption exists, no animals detected, but query IS for an animal
#                 # e.g. "two men in suits" for a "horse" query → exclude
#                 if is_od_match and not is_tag_match and _cap:
#                     _cap_toks_check = {w.rstrip("s") if len(w) > 3 else w
#                                        for w in set(_cap.split())} | set(_cap.split())
#                     _cap_animals_check = _BROAD_ANIMALS & _cap_toks_check
#                     _animal_targets = _coco_targets & {
#                         "cat","dog","horse","bird","cow","elephant","bear",
#                         "zebra","giraffe","sheep","deer","rabbit","horse",
#                     }
#                     if _animal_targets and not _cap_animals_check:
#                         # Caption has NO animal — check if it's a human-only image
#                         _HUMAN_CAP = {"man","woman","person","girl","boy","people","men",
#                                       "women","actor","actress","player","character","he",
#                                       "she","they","guy","lady","child","adult"}
#                         if _HUMAN_CAP & _cap_toks_check:
#                             logger.info(
#                                 f"🚫 OD-filter (human-only caption): id={_img.id} "
#                                 f"caption '{_cap[:60]}' has no animal for {_animal_targets}"
#                             )
#                             continue

#                 # FIX Bug4: non-animal OD targets (car, truck, etc.) with unrelated captions
#                 # e.g. "thor and thor in the avengers movie" for a "red car" query → exclude
#                 if is_od_match and not is_tag_match and _cap:
#                     _VEHICLE_COCO = {"car","truck","bus","motorcycle","bicycle","boat",
#                                      "airplane","train"}
#                     _vehicle_targets = _coco_targets & _VEHICLE_COCO
#                     if _vehicle_targets:
#                         _VEHICLE_WORDS = {"car","truck","bus","motorcycle","bike","bicycle",
#                                           "vehicle","driving","road","highway","parking",
#                                           "train","boat","ship","airplane","plane","suv",
#                                           "sedan","sports","jeep","porsche","ferrari",
#                                           "bmw","mercedes","driving","auto","automobile"}
#                         _FICTIONAL_WORDS = {"marvel","avengers","thor","ironman","iron",
#                                             "spiderman","batman","superman","hulk","anime",
#                                             "cartoon","movie","film","fictional","hero",
#                                             "villain","superhero","comic","bust","armor",
#                                             "avenger","character","suit"}
#                         _cap_set_v = set(_cap.split())
#                         _has_vehicle_word = bool(_VEHICLE_WORDS & _cap_set_v)
#                         _has_fictional    = bool(_FICTIONAL_WORDS & _cap_set_v)
#                         if not _has_vehicle_word and _has_fictional:
#                             logger.info(
#                                 f"🚫 OD-filter (fictional caption for vehicle): id={_img.id} "
#                                 f"caption '{_cap[:60]}'"
#                             )
#                             continue

#                 score = 92.0 if is_tag_match else round(85.0 + _match_score * 5, 1)

#                 # ── BUGFIX: Colour-word filter in OD path ─────────────────────
#                 # The OD fast-path matches by object class (horse/cat/car) but
#                 # completely ignores colour words in the query.
#                 # "white horse" was returning brown/dark horses because OD just
#                 # found all horses and returned them all at score 85–90.
#                 #
#                 # Rules (only when query contains colour words):
#                 #   1. Caption has colour that CONTRADICTS query → EXCLUDE
#                 #      e.g. "brown horse" for "white horse" → skip
#                 #   2. Caption has colour that MATCHES query → KEEP (boosted)
#                 #   3. Caption exists but has NO colour word:
#                 #      → Don't return from OD; fall through to CLIP which uses
#                 #        pixel-level colour scoring (avg_r/g/b) — much more reliable
#                 #        for unlabelled subjects
#                 #   4. No caption yet (BLIP not run) → KEEP (can't verify)
#                 if not is_tag_match and _q_lower_words:
#                     _COLOUR_WORDS = {
#                         "white","black","red","blue","green","yellow","orange",
#                         "purple","pink","brown","gray","grey","golden","silver",
#                         "dark","light","pale","cream","beige","spotted","striped",
#                     }
#                     _q_colours = _COLOUR_WORDS & _q_lower_words
#                     if _q_colours:
#                         _cap_for_colour = (_img.caption_short or "").lower()
#                         if _cap_for_colour:  # only apply filter if caption exists
#                             _cap_toks_c = set(_cap_for_colour.split())
#                             _cap_colours = _COLOUR_WORDS & _cap_toks_c
#                             _matching     = _q_colours & _cap_colours
#                             _contradicting = _cap_colours - _q_colours

#                             # "black and white photo/photograph" = monochrome style,
#                             # not a colour description of the subject → don't exclude
#                             _is_mono = (("black" in _cap_toks_c or "white" in _cap_toks_c) and
#                                         ("photo" in _cap_toks_c or "photograph" in _cap_toks_c
#                                          or "drawing" in _cap_toks_c or "painting" in _cap_toks_c))

#                             if _cap_colours and not _matching and not _is_mono:
#                                 # Caption names a DIFFERENT colour → wrong subject
#                                 logger.info(
#                                     f"🎨 OD colour-filter: id={_img.id} "
#                                     f"query_colours={_q_colours} cap_colours={_cap_colours} → skip"
#                                 )
#                                 continue
#                             elif not _cap_colours:
#                                 # Caption has no colour — defer to CLIP+pixel scoring
#                                 # Don't add to OD results; let it fall through to CLIP
#                                 logger.debug(
#                                     f"🎨 OD colour-defer: id={_img.id} "
#                                     f"no colour in caption → CLIP will score by pixel"
#                                 )
#                                 continue
#                             elif _matching:
#                                 # Confirmed colour match → boost score
#                                 score = min(score + 5, 97.0)

#                 _seen_od.add(_img.id)
#                 _seen_fn.add(fn)
#                 _od_results.append(_od_row(_img, score))

#             # Sort by score desc
#             _od_results.sort(key=lambda x: x["score"], reverse=True)

#             # FIX Bug1: dedup by caption text — catches same-content images uploaded
#             # twice (different UUID filenames, so filename-dedup misses them).
#             _seen_cap_text = set()
#             _od_deduped    = []
#             for _r in _od_results:
#                 _ct = (_r.get("caption_short") or "").strip().lower()
#                 if _ct and _ct in _seen_cap_text:
#                     logger.info(f"🔁 OD dedup: skipping duplicate caption '{_ct[:50]}'")
#                     continue
#                 if _ct:
#                     _seen_cap_text.add(_ct)
#                 _od_deduped.append(_r)
#             _od_results = _od_deduped

#             if _od_results:
#                 logger.info(f"✅ OD-search '{query}' → {len(_od_results)} results via scene_label/tags")
#                 return {
#                     "status": "found",
#                     "query": query,
#                     "count": len(_od_results),
#                     "results": _od_results[:top_k],
#                 }
#         finally:
#             _db_od.close()
#         # OD found nothing → try caption text search for the animal/object words
#         _caption_results = []
#         _seen_cap = set()
#         # Search caption text for query words directly
#         _search_terms = list(_q_lower_words) + [v for k,v in QUERY_TO_COCO.items() if k in _q_lower_words]
#         _search_terms = list(set(_search_terms))
#         _db_cap = SessionLocal()
#         try:
#             for _img in _live(_db_cap).all():
#                 # Search caption, scene_label, AND user tags
#                 _cap_lower = (_img.caption_short or "").lower()
#                 _det_lower = (_img.caption_detailed or "").lower()
#                 _scene_lower = (_img.scene_label or "").lower()
#                 _tags_lower = " ".join(_user_tags(_img)).lower()
#                 _haystack = " ".join(filter(None,[_cap_lower, _det_lower, _scene_lower, _tags_lower]))
#                 # Check if any search term appears anywhere
#                 if _haystack and any(t in _haystack for t in _search_terms) and _img.id not in _seen_cap:
#                     _seen_cap.add(_img.id)
#                     _caption_results.append({
#                         "id": _img.id,
#                         "filename": _img_url(_img.filename),
#                         "score": 80.0,
#                         "timestamp": _img.timestamp.isoformat() if _img.timestamp else None,
#                         "caption_short": _img.caption_short or "",
#                         "person_count": _img.person_count or 0,
#                         "dominant_emotion": _img.dominant_emotion or "",
#                         "quality_level": _img.quality_level or "",
#                         "quality_score": _img.quality_score or 0,
#                         "aesthetic_score": _img.aesthetic_score or 0,
#                         "user_tags": _user_tags(_img),
#                         "photo_note": getattr(_img, "photo_note", "") or "",
#                         "is_favorite": bool(_img.is_favorite),
#                     })
#         finally:
#             _db_cap.close()
#         if _caption_results:
#             logger.info(f"✅ Caption-search '{query}' → {len(_caption_results)} results")
#             return {
#                 "status": "found", "query": query,
#                 "count": len(_caption_results),
#                 "results": sorted(_caption_results, key=lambda x: x["score"], reverse=True)[:top_k]
#             }
#         logger.info(f"⚠️ OD+Caption search found nothing for {_search_terms}, falling back to CLIP")

#     query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
#     if query_emb is None or search_engine.index is None:
#         return {"status": "error", "message": "No images indexed. Upload some photos first."}
#     query_lower  = query.lower()
#     # q_tokens_pre already defined above (before should_search_people pre-compute)

#     # For person-name queries, clear sig_words so CLIP scoring
#     # doesn't give caption bonus to irrelevant images (e.g. pig-in-sink for "Sadie Sink")
#     sig_words    = [] if should_search_people else [w for w in processed_query.lower().split() if len(w) > 2]
#     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in query_lower]

#     # ── Compound word expansion ────────────────────────────────────────────────
#     # BLIP captions write "iron man" (two words) but users search "ironman" (one).
#     # Expand known compounds so all keyword paths find the image.
#     _COMPOUND_SPLITS = {
#         "ironman":     ["iron","man"],      "spiderman":    ["spider","man"],
#         "batman":      ["bat","man"],       "superman":     ["super","man"],
#         "blackwidow":  ["black","widow"],   "antman":       ["ant","man"],
#         "warmachine":  ["war","machine"],   "blackpanther": ["black","panther"],
#         "ironheart":   ["iron","heart"],    "deadpool":     ["dead","pool"],
#         "daredevil":   ["dare","devil"],    "spiderverse":  ["spider","verse"],
#     }
#     if sig_words:
#         _exp = []
#         for _sw in sig_words:
#             _exp.append(_sw)
#             if _sw in _COMPOUND_SPLITS:
#                 _exp.extend(_COMPOUND_SPLITS[_sw])
#         sig_words = list(dict.fromkeys(_exp))

#     # Also expand the processed_query itself so CLIP gets "iron man" instead of "ironman"
#     _pq_words = processed_query.lower().split()
#     _pq_expanded = []
#     for _pw in _pq_words:
#         if _pw in _COMPOUND_SPLITS:
#             _pq_expanded.extend(_COMPOUND_SPLITS[_pw])
#         else:
#             _pq_expanded.append(_pw)
#     if _pq_expanded != _pq_words:
#         processed_query = " ".join(_pq_expanded)
#         logger.info(f"🔤 Compound expand: '{query}' → CLIP query='{processed_query}'")
#     db = SessionLocal()
#     try:
#         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k)
#         # ── Person / actor name search ─────────────────────────────────────
#         # ── Person name search — only for name-like queries ────────────────
#         # CRITICAL: Only run if query looks like a person name, NOT for
#         # descriptive queries like "woman in black dress".
#         # Rules:
#         #   - Must have at least one word >= 4 chars that isn't a common word
#         #   - Must not be a purely descriptive query (too many common words)
#         #   - Person name matching uses WHOLE WORD matching only (not substring)

#         DESCRIPTOR_WORDS = {
#             # articles / prepositions / conjunctions
#             "a","an","the","in","on","at","of","for","with","by","to","from",
#             "into","onto","over","under","above","below","near","next","beside",
#             "and","or","but","not","nor","yet","so","both","either","neither",
#             # verbs
#             "is","are","was","were","be","been","being","has","have","had",
#             "do","does","did","will","would","could","should","may","might",
#             "can","wearing","holding","standing","sitting","posing","looking",
#             "running","walking","smiling","laughing","dancing","jumping","riding",
#             # common adjectives
#             "long","short","tall","small","big","large","little","young","old",
#             "beautiful","pretty","lovely","nice","great","black","white","red",
#             "blue","green","yellow","orange","purple","pink","brown","gray",
#             "dark","light","bright","dark","shiny","fluffy","cute","funny",
#             # people descriptors (not names)
#             "woman","man","girl","boy","lady","guy","person","people","female",
#             "male","human","child","adult","baby","teenager","player","actor",
#             # scene/object words — CRITICAL: prevents these being treated as names
#             "photo","image","picture","standing","next","another","front","back",
#             "window","field","camera","dress","suit","shirt","jacket","hair",
#             "wearing","sitting","holding","looking","posing","smiling",
#             # common household/body nouns that appear in captions
#             "sink","table","chair","floor","wall","door","room","house","home",
#             "hand","face","head","hair","eyes","mouth","nose","body","back",
#             "water","grass","tree","flower","rock","snow","sand","road","street",
#             "book","phone","glass","bowl","plate","ball","box","bag","hat",
#             "food","cake","pizza","coffee","wine","beer","milk","soup",
#             "park","beach","mountain","forest","city","stage","roof","pool",
#             "movie","film","show","song","music","sport","game","team","show",
#             "scene","time","year","day","night","morning","evening","world",
#         }

#         raw_q      = query.lower().strip()
#         q_tokens   = raw_q.split()
#         # Words that could plausibly be part of a name (long, not descriptive)
#         # For short queries (≤3 words like "tom holland"), allow 3-char words
#         min_name_len = 3 if len(q_tokens) <= 3 else 4
#         name_candidates = [
#             w for w in q_tokens
#             if len(w) >= min_name_len
#             and w not in DESCRIPTOR_WORDS
#             and w.isalpha()
#         ]

#         descriptor_count = sum(1 for w in q_tokens if w in DESCRIPTOR_WORDS)
#         is_descriptive   = descriptor_count > len(q_tokens) * 0.5   # strict: >50% descriptors needed
#         should_search_people = bool(name_candidates) and not is_descriptive

#         vector_ids   = {r["id"] for r in results}
#         people_rows  = db.query(Person).all() if should_search_people else []

#         # ── 1. Match against named person clusters ────────────────────────
#         matched_people = []
#         if should_search_people:
#             for p in people_rows:
#                 pname_raw = (p.name or "").strip()
#                 pn        = pname_raw.lower()
#                 if not pn:
#                     continue
#                 # Skip default "Person N" labels
#                 if re.match(r"^person\s+\d+$", pn):
#                     continue
#                 # Skip OCR garbage: all-caps multi-word, or contains digits,
#                 # or contains non-alpha chars (PETRONAS INEQS, Hak IINDIA, etc.)
#                 if re.search(r"\d", pname_raw):
#                     continue   # contains digit → OCR garbage
#                 p_words = pn.split()
#                 # If all words are UPPERCASE in original → likely OCR label not a name
#                 if all(w == w.upper() and len(w) > 1 for w in pname_raw.split()):
#                     continue
#                 # Whole-word matching ONLY — "in" must NOT match inside "india"
#                 # Use word boundary: check if name word appears as standalone token
#                 p_parts = [pt for pt in p_words if len(pt) >= 3]
#                 query_tokens_set = set(q_tokens)
#                 matched = False
#                 for pt in p_parts:
#                     if pt in query_tokens_set:        # exact token match
#                         matched = True; break
#                 for nc in name_candidates:
#                     if nc in p_words:                 # query name-word in person name
#                         matched = True; break
#                     if pn == nc or nc == pn:
#                         matched = True; break
#                     # Allow partial first/last name: "vijay" matches "vijay kumar"
#                     if any(nc == pw for pw in p_words):
#                         matched = True; break
#                 if matched:
#                     matched_people.append(p)

#             # ── 2. Caption / OCR / VQA text search (works WITHOUT renaming) ──
#             # Only run if face-cluster lookup found no results yet
#             # (avoids flooding with false positives from common words in OCR)
#             COMMON_WORDS = {
#                 "the","and","for","are","but","not","you","all","any","can",
#                 "had","her","was","one","our","out","day","get","has","him",
#                 "his","how","its","may","new","now","old","see","two","way",
#                 "who","did","man","men","two","use","she","him","this","that",
#                 "with","from","they","have","more","will","home","also","into",
#                 "over","time","very","when","your","come","here","just","like",
#                 "long","make","many","most","some","them","than","then","these",
#                 "well","were","what","each","much","both","been","only","same",
#                 "india","team","cup","match","player","game","year","last",
#             }
#             # Only do caption search for proper-noun-looking queries
#             # (single words >= 4 chars that aren't common English words)
#             search_words = [w for w in name_candidates if len(w) >= 4 and w not in COMMON_WORDS]
#             if search_words and not matched_people:
#                 for img in _live(db).all():
#                     if img.id in vector_ids:
#                         continue
#                     vqa_person = ""
#                     try:
#                         vqa_d = json.loads(img.caption_vqa or "{}")
#                         vqa_person = vqa_d.get("person", "") or vqa_d.get("subject", "")
#                     except Exception:
#                         pass
#                     hay = " ".join(filter(None, [
#                         img.caption_short or "",
#                         img.caption_detailed or "",
#                         img.ocr_text_enhanced or "",
#                         img.scene_label or "",
#                         vqa_person,
#                         " ".join(_user_tags(img)),
#                     ])).lower()
#                     # Require ALL search words to match (avoids false positives)
#                     if all(w in hay for w in search_words):
#                         results.append({
#                             "id": img.id, "filename": _img_url(img.filename),
#                             "score": 65.0,
#                             "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                             "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
#                             "person_count": img.person_count or 0,
#                             "caption_short": img.caption_short or "",
#                             "user_tags": _user_tags(img),
#                             "photo_note": getattr(img, "photo_note", "") or "",
#                         })
#                         vector_ids.add(img.id)

#             if matched_people:
#                 # CRITICAL: When we found a person match, REPLACE CLIP results entirely.
#                 # CLIP results for name queries are unreliable (e.g. "sadie sink" → pig-in-sink).
#                 # Only return face-cluster matches + caption/OCR name matches.
#                 person_results = []
#                 person_ids_seen = set()

#                 for person in matched_people:
#                     logger.info(f"🧑 Searching for person: '{person.name}' (id={person.id})")
#                     face_records = db.query(DBFace).filter(DBFace.person_id == person.id).all()
#                     face_img_ids = list({f.image_id for f in face_records if f.image_id})
#                     logger.info(f"  → {len(face_records)} face records, {len(face_img_ids)} unique image_ids")

#                     # 1. Face-cluster matched images (highest confidence = 95)
#                     if face_img_ids:
#                         for img in _live(db).filter(DBImage.id.in_(face_img_ids)).all():
#                             if img.id not in person_ids_seen:
#                                 person_results.append({
#                                     "id": img.id, "filename": _img_url(img.filename),
#                                     "score": 95.0,
#                                     "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                                     "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
#                                     "person_count": img.person_count or 0,
#                                     "caption_short": img.caption_short or "",
#                                     "user_tags": _user_tags(img),
#                                     "photo_note": getattr(img, "photo_note", "") or "",
#                                     "matched_person": person.name,
#                                 })
#                                 person_ids_seen.add(img.id)

#                     # 2. Caption/OCR text search for the FULL NAME as a phrase only
#                     # Use full name phrase — prevents "sink" matching pig captions
#                     name_kw = person.name.lower()
#                     name_parts = name_kw.split()  # ["sadie", "sink"]
#                     for img in _live(db).all():
#                         if img.id in person_ids_seen:
#                             continue
#                         hay = " ".join(filter(None, [
#                             img.caption_short or "", img.caption_detailed or "",
#                             img.caption_vqa or "", img.ocr_text_enhanced or "",
#                             img.scene_label or ""
#                         ])).lower()
#                         # Must contain FULL NAME as phrase OR all parts together
#                         # NOT individual words (prevents "sink" matching pig captions)
#                         if name_kw in hay or (len(name_parts) > 1 and all(p in hay for p in name_parts)):
#                             # Extra check: if name has common words (like "sink"),
#                             # require the rare part to appear near human context
#                             person_results.append({
#                                 "id": img.id, "filename": _img_url(img.filename),
#                                 "score": 75.0,
#                                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                                 "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
#                                 "person_count": img.person_count or 0,
#                                 "caption_short": img.caption_short or "",
#                                 "user_tags": _user_tags(img),
#                                 "photo_note": getattr(img, "photo_note", "") or "",
#                                 "matched_person": person.name,
#                             })
#                             person_ids_seen.add(img.id)

#                     if not face_img_ids:
#                         logger.warning(f"  ⚠️ No face records for '{person.name}' — only caption search. Run Re-index AI.")

#                 if person_results:
#                     results = sorted(person_results, key=lambda x: x["score"], reverse=True)
#                     return {"status": "found", "query": query, "count": len(results), "results": results[:top_k]}
#                 else:
#                     # Person found in DB but no photos linked yet — tell user to re-index
#                     names = ", ".join(p.name for p in matched_people)
#                     return {
#                         "status": "not_found",
#                         "message": f"Found person '{names}' but no photos linked. Click Re-index AI to rebuild face clusters.",
#                         "people_suggestions": [{"id": p.id, "name": p.name, "cover": None} for p in matched_people],
#                     }

#         # ── No person found in DB for a name query → try content search first ──
#         if should_search_people and not matched_people:
#             # Before giving up, attempt 3 fallbacks in priority order:
#             # 1. Tag fast-path results (user manually tagged this image)
#             # 2. Caption/OCR keyword search (works for ironman, thor, spider, etc.)
#             # 3. Only if both fail → show "not found" with people suggestions

#             # ── Fallback 1: tags ──────────────────────────────────────────────
#             if _tag_fast_results:
#                 logger.info(f"🏷️ Person-path fallback → {len(_tag_fast_results)} tag results for '{query}'")
#                 return {
#                     "status": "found", "query": query,
#                     "count": len(_tag_fast_results),
#                     "results": _tag_fast_results[:top_k],
#                 }

#             # ── Fallback 2: caption + OCR keyword search ──────────────────────
#             # Covers: "ironman", "thor", "spider", "kpop", "user" (OCR), etc.
#             # Use the original query words (NOT processed_query which strips non-CLIP terms)
#             _kw_query_words = [w for w in query.lower().split() if len(w) > 2]
#             _kw_results_fb  = []
#             _kw_seen_fb     = set()
#             _kw_seen_cap_fb = set()
#             if _kw_query_words:
#                 for _img_fb in _live(db).all():
#                     if _img_fb.id in _kw_seen_fb:
#                         continue
#                     # Build search haystack: caption + OCR + scene_label + tags
#                     _fb_hay = " ".join(filter(None, [
#                         (_img_fb.caption_short    or "").lower(),
#                         (_img_fb.caption_detailed or "").lower(),
#                         (_img_fb.ocr_text_enhanced or "").lower(),
#                         (_img_fb.scene_label      or "").lower(),
#                         " ".join(_user_tags(_img_fb)).lower(),
#                     ]))
#                     if not _fb_hay:
#                         continue
#                     # ALL query words must appear (strict — avoids false positives)
#                     if all(w in _fb_hay for w in _kw_query_words):
#                         _ct_fb = (_img_fb.caption_short or "").strip().lower()
#                         if _ct_fb and _ct_fb in _kw_seen_cap_fb:
#                             continue
#                         if _ct_fb:
#                             _kw_seen_cap_fb.add(_ct_fb)
#                         _kw_seen_fb.add(_img_fb.id)
#                         _kw_results_fb.append({
#                             "id": _img_fb.id,
#                             "filename": _img_url(_img_fb.filename) or "",
#                             "score": 80.0,
#                             "timestamp": _img_fb.timestamp.isoformat() if _img_fb.timestamp else None,
#                             "caption_short": _img_fb.caption_short or "",
#                             "person_count": _img_fb.person_count or 0,
#                             "dominant_emotion": _img_fb.dominant_emotion or "",
#                             "quality_level": _img_fb.quality_level or "",
#                             "quality_score": _img_fb.quality_score or 0,
#                             "aesthetic_score": _img_fb.aesthetic_score or 0,
#                             "user_tags": _user_tags(_img_fb),
#                             "photo_note": getattr(_img_fb, "photo_note", "") or "",
#                             "is_favorite": bool(_img_fb.is_favorite),
#                             "scene_label": _img_fb.scene_label or "",
#                             "width": _img_fb.width, "height": _img_fb.height,
#                         })
#             if _kw_results_fb:
#                 logger.info(f"📝 Person-path caption/OCR fallback → {len(_kw_results_fb)} results for '{query}'")
#                 return {
#                     "status": "found", "query": query,
#                     "count": len(_kw_results_fb),
#                     "results": sorted(_kw_results_fb, key=lambda x: x["score"], reverse=True)[:top_k],
#                 }

#             # ── Fallback 3: truly nothing found → people suggestions ──────────
#             all_people = db.query(Person).all()
#             suggestions = []
#             for p in all_people:
#                 if re.match(r"^person\s+\d+$", (p.name or "").lower()): continue
#                 face_recs = db.query(DBFace).filter(DBFace.person_id == p.id).limit(1).all()
#                 if face_recs and face_recs[0].image_id:
#                     cover_img = db.query(DBImage).filter(DBImage.id == face_recs[0].image_id).first()
#                     if cover_img:
#                         suggestions.append({"id": p.id, "name": p.name, "cover": _img_url(cover_img.filename)})
#             return {
#                 "status": "not_found",
#                 "message": f"No photos matched '{query}'.",
#                 "people_suggestions": suggestions[:12],
#             }

#         # Fallback: discard any CLIP junk when person matched
#         if matched_people:
#             person_results = [r for r in results if r.get("matched_person")]
#             if person_results:
#                 results = person_results
#             results.sort(key=lambda x: x["score"], reverse=True)

#         # ── Skip keyword fallback when we already have person results ──────────
#         # Person name searches should ONLY return person face images, not
#         # generic CLIP results or keyword matches for ambiguous words like "sink"
#         skip_kw_fallback = bool(matched_people)

#         if sig_words and not skip_kw_fallback:
#             vector_ids = {r["id"] for r in results}
#             KW_STOP = {"the","a","an","in","on","at","of","for","with","by","to",
#                        "from","is","are","was","were","be","and","or","but","not"}
#             meaningful = [w for w in sig_words if w not in KW_STOP and len(w) > 2]

#             # ── Rebuild contradiction context for filtering ────────────────
#             COLOURS_KW = {"black","white","red","blue","green","yellow","orange",
#                           "purple","pink","brown","gray","grey","golden","silver"}
#             CLOTHING_KW = {"dress","suit","shirt","jacket","sari","skirt","gown",
#                            "uniform","coat","blouse","saree","lehenga"}
#             ANIMALS_KW = {"horse","cow","dog","cat","bird","fox","tiger","lion",
#                           "bear","wolf","pig","rabbit","fish","duck","frog","otter",
#                           "elephant","monkey","deer","sheep","goat","snake","chicken",
#                           "kitten","puppy","calf","hamster","squirrel","raccoon"}
#             FEMALE_KW = {"woman","girl","lady","female"}
#             MALE_KW   = {"man","boy","guy","male"}
#             q_set_kw      = set(meaningful)
#             q_colours_kw  = COLOURS_KW  & q_set_kw
#             q_clothing_kw = CLOTHING_KW & q_set_kw
#             q_animals_kw  = ANIMALS_KW  & q_set_kw
#             q_gender_kw   = (FEMALE_KW | MALE_KW) & q_set_kw

#             def _kw_ok(caption_text):
#                 """Return True if caption doesn't contradict the query."""
#                 cap = set(caption_text.split())
#                 # Stemmed cap words
#                 def s(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
#                 cap_stem = {s(w) for w in cap} | cap

#                 if q_animals_kw:
#                     cap_animals = ANIMALS_KW & cap_stem
#                     if cap_animals and not (cap_animals & q_animals_kw):
#                         return False   # different animal
#                     # Also: if caption has PEOPLE words but NO animals at all → exclude
#                     # (e.g. "man standing in a field" shouldn't appear for "dog" query)
#                     PEOPLE_KW = {"man","woman","person","girl","boy","people","guy",
#                                  "lady","male","female","player","actor","celebrity"}
#                     if not cap_animals and (PEOPLE_KW & cap_stem):
#                         return False   # people-only caption for animal query
#                 if q_colours_kw:
#                     cap_col = COLOURS_KW & cap
#                     contradicting = cap_col - q_colours_kw
#                     matching      = cap_col & q_colours_kw
#                     if contradicting and not matching:
#                         return False   # only wrong colours
#                 if q_clothing_kw:
#                     cap_cl = CLOTHING_KW & cap_stem
#                     if cap_cl and not (cap_cl & q_clothing_kw):
#                         return False   # different clothing
#                 if q_gender_kw:
#                     q_f = bool(FEMALE_KW & q_gender_kw)
#                     q_m = bool(MALE_KW   & q_gender_kw)
#                     if q_f and (MALE_KW & cap) and not (FEMALE_KW & cap):
#                         return False
#                     if q_m and (FEMALE_KW & cap) and not (MALE_KW & cap):
#                         return False
#                 return True

#             kw_candidates = _live(db).filter(
#                 DBImage.caption_short.isnot(None) | DBImage.caption_detailed.isnot(None)
#             ).all()
#             for img in kw_candidates:
#                 if img.id in vector_ids:
#                     continue
#                 caption_text = " ".join(filter(None,[
#                     img.caption_short or "", img.caption_detailed or ""
#                 ])).lower()
#                 haystack = caption_text + " " + " ".join(filter(None,[
#                     img.ocr_text_enhanced or "", img.scene_label or "",
#                     " ".join(_user_tags(img))
#                 ])).lower()
#                 if not meaningful:
#                     continue
#                 # Apply contradiction filter FIRST
#                 if not _kw_ok(caption_text):
#                     continue
#                 # Compound word matching: "ironman" counts as matched if both
#                 # "iron" AND "man" are present (even as separate words)
#                 def _word_in_haystack(w, hs):
#                     if w in hs:
#                         return True
#                     parts = _COMPOUND_SPLITS.get(w)
#                     if parts and all(p in hs for p in parts):
#                         return True
#                     return False
#                 # For 1-2 meaningful words: ALL must match
#                 # For 3+: majority (≥70%) must match
#                 matched = sum(1 for w in meaningful if _word_in_haystack(w, haystack))
#                 threshold = len(meaningful) if len(meaningful) <= 2 else max(2, len(meaningful) * 0.70)
#                 if matched >= threshold:
#                     results.append({
#                         "id": img.id, "filename": _img_url(img.filename), "score": 32.0,
#                         "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                         "person_count": img.person_count or 0,
#                         "caption_short": img.caption_short or "",
#                         "caption_detailed": img.caption_detailed or "",
#                         "quality_level": img.quality_level or "",
#                         "quality_score": img.quality_score or 0,
#                         "dominant_emotion": img.dominant_emotion or "",
#                         "face_emotion_count": img.face_emotion_count or 0,
#                         "aesthetic_score": img.aesthetic_score or 0,
#                         "aesthetic_rating": img.aesthetic_rating or "",
#                         "is_favorite": bool(img.is_favorite),
#                         "user_tags": _user_tags(img),
#                         "photo_note": getattr(img, "photo_note", "") or "",
#                         "scene_label": img.scene_label or "",
#                         "width": img.width, "height": img.height,
#                     })
#             results.sort(key=lambda x: x["score"], reverse=True)
#             results = results[:top_k]

#         elif sig_words and skip_kw_fallback:
#             # Person search mode: drop any CLIP results that aren't person matches
#             # (e.g. "sadie sink" → remove literal sink images from CLIP)
#             person_img_ids = {r["id"] for r in results if r.get("matched_person")}
#             if person_img_ids:
#                 results = [r for r in results if r.get("matched_person") or r["id"] in person_img_ids]
#             results.sort(key=lambda x: x["score"], reverse=True)
#             results = results[:top_k]
#         if not results:
#             # If CLIP found nothing but the tag fast-path did, return tag results directly
#             if _tag_fast_results:
#                 logger.info(f"🏷️ CLIP found nothing — returning {len(_tag_fast_results)} tag-matched results")
#                 return {
#                     "status": "found", "query": query,
#                     "count": len(_tag_fast_results),
#                     "results": _tag_fast_results[:top_k],
#                 }
#             # Return people suggestions so user can identify who to search
#             people_suggestions = []
#             if name_candidates:
#                 all_people = db.query(Person).all()
#                 for p in all_people:
#                     face_recs = db.query(DBFace).filter(DBFace.person_id == p.id).limit(1).all()
#                     if face_recs and face_recs[0].image_id:
#                         cover_img = db.query(DBImage).filter(DBImage.id == face_recs[0].image_id).first()
#                         if cover_img:
#                             people_suggestions.append({
#                                 "id": p.id,
#                                 "name": p.name,
#                                 "cover": _img_url(cover_img.filename)
#                             })
#             return {
#                 "status": "not_found",
#                 "message": f"No images matched '{query}'.",
#                 "people_suggestions": people_suggestions[:12],
#             }
#         # ── Merge tag fast-path results (score=97) before dedup ─────────────
#         # Tag-matched images that weren't already in CLIP results get prepended
#         # at the top so they always appear first regardless of CLIP score.
#         if _tag_fast_results:
#             existing_ids = {r["id"] for r in results}
#             for _tr in _tag_fast_results:
#                 if _tr["id"] not in existing_ids:
#                     results.insert(0, _tr)
#                     existing_ids.add(_tr["id"])
#                 else:
#                     # Already in results — boost its score to 97
#                     for _r in results:
#                         if _r["id"] == _tr["id"]:
#                             _r["score"] = max(_r.get("score", 0), 97.0)
#                             break
#             results.sort(key=lambda x: x["score"], reverse=True)

#         # ── BUGFIX: Final dedup by filename ─────────────────────────────────
#         # Catches re-indexed duplicate DB rows (different id, same physical file)
#         # that bypass the earlier id-based dedup in _score_candidates.
#         _seen_fn_final  = set()
#         _seen_cap_final = set()   # also dedup by caption — catches same-content re-uploads
#         _deduped_final  = []
#         for _r in results:
#             _fn  = _r.get("filename") or ""
#             _ct  = (_r.get("caption_short") or "").strip().lower()
#             if _fn and _fn in _seen_fn_final:
#                 continue
#             # Caption dedup: skip if identical caption already in results
#             # (keep first = highest scored; favourites get boosted by tag merge so appear first)
#             if _ct and _ct in _seen_cap_final:
#                 continue
#             if _fn:
#                 _seen_fn_final.add(_fn)
#             if _ct:
#                 _seen_cap_final.add(_ct)
#             _deduped_final.append(_r)
#         results = _deduped_final

#         return {"status": "found", "query": query, "count": len(results), "results": results}
#     finally:
#         db.close()

# @app.post("/search/describe")
# def search_by_description(description: str = Form(...), top_k: int = Form(20)):
#     if not description or not description.strip():
#         return {"status": "error", "message": "Description empty"}
#     if search_engine.index is None:
#         return {"status": "error", "message": "No images indexed."}
#     desc = _clean_query(description.strip())
#     prompts = [desc, f"a photo of {desc}", f"an image showing {desc}", f"a picture of {desc}", f"{desc} photograph"]
#     embs = [e for p in prompts for e in [search_engine.get_text_embedding(p, use_prompt_ensemble=False)] if e is not None]
#     if not embs:
#         return {"status": "error", "message": "Could not encode description."}
#     query_emb = np.mean(embs, axis=0).astype("float32")
#     norm = np.linalg.norm(query_emb)
#     if norm > 1e-8:
#         query_emb /= norm
#     sig_words    = [w for w in desc.lower().split() if len(w) > 2]
#     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in desc.lower()]
#     db = SessionLocal()
#     try:
#         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k)
#         if not results:
#             return {"status": "not_found", "message": f"No images matched description '{desc}'."}
#         return {"status": "found", "query": desc, "count": len(results), "results": results}
#     finally:
#         db.close()

# @app.post("/search/hybrid")
# async def search_hybrid(query: str = Form(""), file: UploadFile = File(None), text_weight: float = Form(0.6), image_weight: float = Form(0.4), top_k: int = Form(20)):
#     if search_engine.index is None:
#         return {"status": "error", "message": "No images indexed."}
#     has_text  = query.strip() != ""
#     has_image = file is not None and file.filename
#     if not has_text and not has_image:
#         return {"status": "error", "message": "Provide at least a query or an image."}
#     if has_text and has_image:
#         total_w = text_weight + image_weight
#         if total_w <= 0:
#             text_weight, image_weight = 0.6, 0.4
#             total_w = 1.0
#         text_weight /= total_w; image_weight /= total_w
#     elif has_text:
#         text_weight, image_weight = 1.0, 0.0
#     else:
#         text_weight, image_weight = 0.0, 1.0
#     text_emb = None
#     if has_text:
#         text_emb = search_engine.get_text_embedding(_clean_query(query), use_prompt_ensemble=True)
#     img_emb = None
#     if has_image:
#         import tempfile
#         ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
#         with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
#             tmp_path = tmp.name
#             shutil.copyfileobj(file.file, tmp)
#         try:
#             img_emb = search_engine.get_image_embedding(tmp_path)
#         finally:
#             try: os.remove(tmp_path)
#             except Exception: pass
#     if text_emb is not None and img_emb is not None:
#         query_emb, extra_emb = text_emb, img_emb
#     elif text_emb is not None:
#         query_emb, extra_emb = text_emb, None
#     elif img_emb is not None:
#         query_emb, extra_emb = img_emb, None; text_weight, image_weight = 1.0, 0.0
#     else:
#         return {"status": "error", "message": "Could not encode query."}
#     sig_words    = [w for w in query.lower().split() if len(w) > 2]
#     query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in query.lower()]
#     db = SessionLocal()
#     try:
#         results = _score_candidates(query_emb, sig_words, query_colors, db, top_k, extra_emb=extra_emb, text_weight=text_weight, image_weight=image_weight)
#         if not results:
#             return {"status": "not_found", "message": "No images matched."}
#         return {"status": "found", "query": query, "count": len(results), "text_weight": text_weight, "image_weight": image_weight, "results": results}
#     finally:
#         db.close()

# @app.post("/search/voice")
# def voice_search_legacy(duration: int = Form(5)):
#     try:
#         transcribed = voice_engine.listen_and_transcribe(duration=duration)
#         if not transcribed or not transcribed.strip():
#             return {"status": "error", "message": "Could not hear anything."}
#         result = search(query=transcribed.strip(), top_k=20)
#         result["transcribed"] = transcribed.strip()
#         return result
#     except Exception as e:
#         return {"status": "error", "message": f"Voice search failed: {str(e)}"}

# @app.post("/search/image")
# async def search_by_image(file: UploadFile = File(...), top_k: int = Form(20)):
#     if search_engine.index is None:
#         return {"status": "error", "message": "No images indexed."}
#     ext = os.path.splitext(file.filename or "")[1].lower()
#     if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
#         raise HTTPException(status_code=400, detail="Unsupported format.")
#     import tempfile
#     with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
#         tmp_path = tmp.name
#         shutil.copyfileobj(file.file, tmp)
#     try:
#         query_emb = search_engine.get_image_embedding(tmp_path)
#     finally:
#         try: os.remove(tmp_path)
#         except Exception: pass
#     if query_emb is None:
#         return {"status": "error", "message": "Could not process image."}
#     total = search_engine.index.ntotal
#     q = query_emb.reshape(1, -1).astype("float32")
#     faiss.normalize_L2(q)
#     distances, indices = search_engine.index.search(q, min(top_k * 3, total))
#     db = SessionLocal()
#     try:
#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx == -1: continue
#             score = float(dist)
#             if score < 0.45: break
#             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
#             if not img or img.is_trashed: continue
#             results.append({"id": img.id, "filename": _img_url(img.filename), "score": round(score * 100, 2), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None, "person_count": img.person_count or 0})
#         results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
#         if not results:
#             return {"status": "not_found", "message": "No visually similar images found."}
#         return {"status": "found", "count": len(results), "results": results}
#     finally:
#         db.close()

# @app.post("/search/color")
# def search_by_color(color: str = Form(...), top_k: int = Form(20)):
#     """
#     Color search using HSV hue matching instead of raw RGB distance.
#     Much more accurate — a blue sky photo scores high for 'blue' regardless
#     of how many green trees are also in the frame.
#     """
#     import colorsys

#     # ── Hue ranges in degrees (0-360) for each color ────────────────────────
#     # Each entry: (hue_center, hue_half_width, min_saturation, min_value)
#     COLOR_HSV = {
#         "red":    [(  0, 18, 0.35, 0.20), (360, 18, 0.35, 0.20)],  # red wraps around 0/360
#         "orange": [( 25, 15, 0.45, 0.25)],
#         "yellow": [( 55, 18, 0.40, 0.30)],
#         "green":  [(120, 40, 0.30, 0.15)],
#         "blue":   [(220, 45, 0.30, 0.15)],
#         "purple": [(280, 30, 0.25, 0.15)],
#         "pink":   [(330, 25, 0.25, 0.30)],
#         "white":  None,   # special: high value, low saturation
#         "black":  None,   # special: low value
#         "gray":   None,   # special: low saturation
#         "grey":   None,
#         "brown":  [( 25, 18, 0.35, 0.15)],  # like orange but lower value
#     }

#     color_key = color.strip().lower()
#     if color_key not in COLOR_HSV:
#         color_key = next((k for k in COLOR_HSV if k in color_key), None)
#     if not color_key:
#         return {"status": "error", "message": f"Unknown color '{color}'."}

#     def _color_score(r, g, b, ckey):
#         """Score 0-100 how well an RGB pixel matches the target color."""
#         r_, g_, b_ = (r or 0)/255.0, (g or 0)/255.0, (b or 0)/255.0
#         h, s, v = colorsys.rgb_to_hsv(r_, g_, b_)
#         hue_deg = h * 360.0

#         if ckey in ("white", "grey", "gray"):
#             # High brightness, low saturation
#             if ckey == "white":
#                 return max(0.0, (v - 0.75) / 0.25 * 100) * max(0.0, (0.25 - s) / 0.25)
#             else:  # gray
#                 sat_score = max(0.0, (0.20 - s) / 0.20)
#                 val_score = max(0.0, 1.0 - abs(v - 0.50) / 0.50)
#                 return sat_score * val_score * 100
#         if ckey == "black":
#             return max(0.0, (0.25 - v) / 0.25 * 100)

#         ranges = COLOR_HSV[ckey]
#         if not ranges:
#             return 0.0

#         best = 0.0
#         for entry in ranges:
#             hc, hw, min_s, min_v = entry
#             # Hue distance (circular)
#             diff = abs(hue_deg - hc)
#             if diff > 180: diff = 360 - diff
#             if diff > hw * 2:
#                 continue
#             hue_score  = max(0.0, 1.0 - diff / (hw * 2))
#             sat_score  = min(1.0, max(0.0, (s - min_s) / (1.0 - min_s + 0.01)))
#             val_score  = min(1.0, max(0.0, (v - min_v) / (1.0 - min_v + 0.01)))
#             # brown special case: penalise high-value (bright) oranges
#             if ckey == "brown":
#                 val_score *= max(0.0, 1.0 - max(0.0, v - 0.55) / 0.45)
#             score = hue_score * sat_score * val_score
#             best = max(best, score)
#         return round(best * 100, 2)

#     db = SessionLocal()
#     try:
#         images = _live(db).filter(
#             DBImage.avg_r != None, DBImage.avg_g != None, DBImage.avg_b != None
#         ).all()
#         if not images:
#             return {"status": "not_found", "message": "No color data. Upload photos first."}

#         scored = []
#         for img in images:
#             s = _color_score(img.avg_r, img.avg_g, img.avg_b, color_key)
#             if s >= 10.0:  # low threshold — HSV matching is already precise
#                 scored.append((s, img))

#         scored.sort(key=lambda x: x[0], reverse=True)
#         scored = scored[:top_k]

#         if not scored:
#             return {"status": "not_found", "message": f"No images matched color '{color}'."}
#         return {
#             "status": "found", "query": color, "count": len(scored),
#             "results": [{
#                 "id": img.id, "filename": _img_url(img.filename),
#                 "score": round(s, 2),
#                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                 "caption_short": img.caption_short or "",
#                 "user_tags": _user_tags(img),
#                 "dominant_emotion": img.dominant_emotion,
#                 "person_count": img.person_count or 0,
#             } for s, img in scored]
#         }
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # REPROCESS EMOTIONS (fixes existing photos that have "neutral" default)
# # ────────────────────────────────────────────────────────────────────────────
# @app.post("/reprocess-colors")
# def reprocess_colors():
#     """Recompute dominant color for all existing photos using weighted histogram."""
#     import colorsys as _cs2
#     from PIL import Image as PILImage2
#     db = SessionLocal()
#     try:
#         images = _live(db).all()
#         done = 0
#         for img in images:
#             fpath = img.original_path or os.path.join(IMAGE_DIR, img.filename)
#             if not os.path.exists(fpath):
#                 continue
#             try:
#                 pil = PILImage2.open(fpath).convert("RGB").resize((64,64))
#                 arr2 = np.array(pil, dtype=np.float32)
#                 r_v = arr2[:,:,0].flatten(); g_v = arr2[:,:,1].flatten(); b_v = arr2[:,:,2].flatten()
#                 wts = np.array([
#                     _cs2.rgb_to_hsv(r/255,g/255,b/255)[1] * _cs2.rgb_to_hsv(r/255,g/255,b/255)[2] + 0.01
#                     for r,g,b in zip(r_v,g_v,b_v)
#                 ])
#                 tw = wts.sum()
#                 img.avg_r = float(np.dot(r_v, wts)/tw)
#                 img.avg_g = float(np.dot(g_v, wts)/tw)
#                 img.avg_b = float(np.dot(b_v, wts)/tw)
#                 done += 1
#             except Exception as e:
#                 logger.warning(f"Color reprocess failed {img.id}: {e}")
#         db.commit()
#         return {"status": "done", "updated": done}
#     except Exception as e:
#         db.rollback(); raise HTTPException(500, str(e))
#     finally:
#         db.close()


# @app.post("/reprocess-emotions")
# async def reprocess_emotions(background_tasks: BackgroundTasks, limit: int = 0):
#     """
#     Re-detect emotions on all uploaded images that still have the default
#     'neutral' label or no emotion set. Runs in background so the API returns
#     immediately.  limit=0 means process ALL.
#     """
#     db = SessionLocal()
#     try:
#         q = _live(db)
#         if limit > 0:
#             q = q.limit(limit)
#         image_ids = [
#             (img.id, img.original_path or os.path.join(IMAGE_DIR, img.filename))
#             for img in q.all()
#         ]
#     finally:
#         db.close()

#     def _run_reprocess(pairs):
#         done = 0
#         for img_id, fpath in pairs:
#             if not os.path.exists(fpath):
#                 continue
#             try:
#                 # ── GATE: use InsightFace face count, not Faster R-CNN person_count ──
#                 # person_count = 0 for close-up portraits (no full body visible).
#                 # InsightFace accurately detects human faces regardless of body visibility.
#                 # Cat faces, car headlights, turtles → InsightFace gives 0 → skip model.
#                 _db_chk = SessionLocal()
#                 try:
#                     _face_count = _db_chk.query(DBFace).filter(DBFace.image_id == img_id).count()
#                 except Exception:
#                     _face_count = -1  # unknown → let it through
#                 finally:
#                     _db_chk.close()

#                 if _face_count == 0:
#                     # No real human faces → clear any previously wrong emotion label
#                     _db_n = SessionLocal()
#                     try:
#                         _db_n.query(DBImage).filter(DBImage.id == img_id).update({
#                             "dominant_emotion": "neutral", "face_emotion_count": 0,
#                             "emotion_data": "[]",
#                         }, synchronize_session=False)
#                         _db_n.commit()
#                     finally:
#                         _db_n.close()
#                     continue

#                 ed = emotion_detection.detect_emotions(fpath)
#                 # Confidence filter — same as _enrich_image
#                 ed = [e for e in ed if e.get("confidence", 0) >= 0.50]
#                 dominant = ed[0]["emotion"] if ed else "neutral"
#                 db2 = SessionLocal()
#                 try:
#                     db2.query(DBImage).filter(DBImage.id == img_id).update({
#                         "dominant_emotion":   dominant,
#                         "face_emotion_count": len(ed),
#                         "emotion_data":       _json.dumps(ed),
#                     }, synchronize_session=False)
#                     db2.commit()
#                     done += 1
#                 finally:
#                     db2.close()
#             except Exception as e:
#                 logger.warning(f"Emotion reprocess failed for {img_id}: {e}")
#         logger.info(f"✅ Emotion reprocess complete: {done}/{len(pairs)} images updated")

#     background_tasks.add_task(_run_reprocess, image_ids)
#     return {
#         "status": "started",
#         "message": f"Reprocessing emotions for {len(image_ids)} images in background",
#         "count": len(image_ids)
#     }


# @app.post("/reprocess-names")
# def reprocess_names():
#     """
#     Re-run auto-naming on ALL existing person clusters.
#     Resets garbage names (ALL-CAPS brands, OCR fragments, short initials)
#     back to "Person N", then re-runs the improved auto-naming logic.
#     """
#     db = SessionLocal()
#     try:
#         # ── Step 1: Reset bad auto-assigned names ───────────────────────────
#         people_all = db.query(Person).all()
#         reset_count = 0
#         for i, p in enumerate(people_all):
#             pn = (p.name or "").strip()
#             # Never touch manually typed names (user renamed them — they know best)
#             # We can't tell manual from auto, so use heuristics to detect garbage:
#             should_reset = False

#             if not pn or len(pn) < 2:
#                 should_reset = True
#             elif not any(c.isalpha() for c in pn):
#                 should_reset = True
#             else:
#                 name_words = pn.split()
#                 # Check every word in the name
#                 bad_word_count = 0
#                 for nw in name_words:
#                     nw_clean = nw.rstrip(".,!?;:")
#                     # ALL-CAPS word (brand name / jersey text)
#                     if nw_clean.isalpha() and nw_clean == nw_clean.upper() and len(nw_clean) > 2:
#                         bad_word_count += 1
#                     # Too short (< 3 chars = initials / OCR fragment)
#                     elif len(nw_clean) < 3:
#                         bad_word_count += 1
#                     # Known garbage word
#                     elif nw_clean.lower() in _BLIP_NON_NAMES:
#                         bad_word_count += 1
#                     # No vowels (pure consonant = OCR garbage)
#                     elif not any(c in "aeiouAEIOU" for c in nw_clean):
#                         bad_word_count += 1
#                 # If ANY word in the name is garbage → reset the whole name
#                 if bad_word_count > 0:
#                     should_reset = True

#             if should_reset:
#                 p.name = f"Person {i + 1}"
#                 reset_count += 1

#         db.commit()
#         logger.info(f"🔄 Reset {reset_count} bad auto-names back to defaults")

#         # ── Step 2: Re-run auto-naming with improved logic ──────────────────
#         people_all = db.query(Person).all()
#         person_map = {i: p.id for i, p in enumerate(people_all)}
#         _auto_name_people(db, person_map)
#         db.commit()

#         named = db.query(Person).filter(~Person.name.startswith("Person ")).count()
#         total = db.query(Person).count()
#         return {
#             "status": "done",
#             "reset": reset_count,
#             "named": named,
#             "total": total,
#             "message": f"{named}/{total} people named · {reset_count} bad names reset"
#         }
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(500, str(e))
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # FACES / PEOPLE
# # ────────────────────────────────────────────────────────────────────────────
# @app.get("/faces")
# def get_faces(person_id: int = Query(None)):
#     db = SessionLocal()
#     try:
#         if person_id:
#             person = db.query(Person).filter(Person.id == person_id).first()
#             if not person:
#                 raise HTTPException(status_code=404)
#             rows = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == person_id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
#             cover  = _img_url(rows[0].filename) if rows else None
#             images = [{"id": img.id, "filename": _img_url(img.filename), "thumbnail": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None} for img in rows]
#             return {"id": person.id, "name": person.name, "face_count": len(images), "cover": cover, "images": images, "results": images}
#         else:
#             people  = db.query(Person).all()
#             results = []
#             for p in people:
#                 imgs = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == p.id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
#                 if not imgs: continue
#                 results.append({"id": p.id, "name": p.name, "count": len(imgs), "cover": _img_url(imgs[0].filename)})
#             return {"results": results, "count": len(results)}
#     finally:
#         db.close()

# @app.get("/people/search")
# def search_people_by_name(q: str = Query(...)):
#     """Search people by name — returns matching persons with their photos."""
#     db = SessionLocal()
#     try:
#         q_lower = q.lower().strip()
#         if not q_lower:
#             return {"results": []}
#         all_people = db.query(Person).all()
#         matched = []
#         for p in all_people:
#             pn = (p.name or "").lower().strip()
#             if not pn or pn in ("unknown",):
#                 continue
#             parts = pn.split()
#             if q_lower in pn or any(q_lower in part for part in parts) or any(part in q_lower for part in parts if len(part)>=2):
#                 face_records = db.query(DBFace).filter(DBFace.person_id == p.id).all()
#                 img_ids = list({f.image_id for f in face_records if f.image_id})
#                 imgs = _live(db).filter(DBImage.id.in_(img_ids)).order_by(DBImage.timestamp.desc()).all() if img_ids else []
#                 cover = _img_url(imgs[0].filename) if imgs else None
#                 matched.append({
#                     "id": p.id, "name": p.name, "count": len(imgs), "cover": cover,
#                     "results": [{"id": img.id, "filename": _img_url(img.filename),
#                                  "score": 92.0, "matched_person": p.name,
#                                  "caption_short": img.caption_short or "",
#                                  "timestamp": img.timestamp.isoformat() if img.timestamp else None,
#                                  "user_tags": _user_tags(img),
#                                  "person_count": img.person_count or 0} for img in imgs]
#                 })
#         return {"query": q, "count": len(matched), "results": matched}
#     finally:
#         db.close()


# @app.get("/people/{person_id}")
# def get_person(person_id: int):
#     db = SessionLocal()
#     try:
#         person = db.query(Person).filter(Person.id == person_id).first()
#         if not person:
#             raise HTTPException(status_code=404)
#         rows = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == person_id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
#         cover  = _img_url(rows[0].filename) if rows else None
#         images = [{"id": img.id, "filename": _img_url(img.filename), "thumbnail": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None} for img in rows]
#         return {"id": person.id, "name": person.name, "face_count": len(images), "cover": cover, "images": images, "results": images}
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
#     return {"status": "no_match"}


# # ────────────────────────────────────────────────────────────────────────────
# # ALBUMS
# # ────────────────────────────────────────────────────────────────────────────
# @app.get("/albums/{album_id}")
# def get_album_by_id(album_id: int):
#     db = SessionLocal()
#     try:
#         album = db.query(Album).filter(Album.id == album_id).first()
#         if not album:
#             raise HTTPException(status_code=404, detail=f"Album {album_id} not found")
#         images = _live(db).filter(DBImage.album_id == album_id).order_by(DBImage.timestamp).all()
#         cover  = _img_url(images[0].filename) if images else None
#         date_str = ""
#         if album.start_date:
#             date_str = album.start_date.strftime("%b %Y")
#             if album.end_date and album.end_date.month != album.start_date.month:
#                 date_str += f" – {album.end_date.strftime('%b %Y')}"
#         img_list = [{"id": img.id, "filename": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None, "caption_short": img.caption_short, "ocr_text_enhanced": img.ocr_text_enhanced, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "aesthetic_score": img.aesthetic_score} for img in images]
#         return {"id": album.id, "title": album.title, "type": album.type, "description": album.description, "date": date_str, "cover": cover, "start_date": album.start_date.isoformat() if album.start_date else None, "end_date": album.end_date.isoformat() if album.end_date else None, "image_count": len(images), "images": img_list, "results": img_list, "thumbnails": [_img_url(img.filename) for img in images[:4]]}
#     finally:
#         db.close()

# @app.get("/albums")
# def get_albums(album_id: int = Query(None)):
#     db = SessionLocal()
#     try:
#         if album_id:
#             return get_album_by_id(album_id)
#         albums  = db.query(Album).all()
#         results = []
#         for a in albums:
#             album_images = _live(db).filter(DBImage.album_id == a.id).all()
#             # Show empty manual albums — user just created them
#             if not album_images and a.type == "event":
#                 continue  # hide empty auto-event albums, show empty manual ones
#             date_str = ""
#             if a.start_date:
#                 date_str = a.start_date.strftime("%b %Y")
#                 if a.end_date and a.end_date.month != a.start_date.month:
#                     date_str += f" – {a.end_date.strftime('%b %Y')}"
#             cover = _img_url(album_images[0].filename) if album_images else None
#             results.append({"id": a.id, "title": a.title, "type": a.type or "manual",
#                             "description": a.description, "date": date_str,
#                             "cover": cover, "count": len(album_images),
#                             "thumbnails": [_img_url(img.filename) for img in album_images[:4]]})
#         return {"results": results, "count": len(results)}
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # ALBUM CRUD
# # ────────────────────────────────────────────────────────────────────────────

# @app.post("/albums/create")
# def create_album(
#     title: str = Form(...),
#     description: str = Form(""),
#     image_ids: str = Form(""),   # comma-separated image IDs to add immediately
# ):
#     """Create a new manual album, optionally with a set of images."""
#     db = SessionLocal()
#     try:
#         album = Album(
#             title=title.strip(),
#             description=description.strip() or None,
#             type="manual",
#         )
#         db.add(album); db.flush()

#         added = 0
#         if image_ids.strip():
#             ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
#             for img in _live(db).filter(DBImage.id.in_(ids)).all():
#                 img.album_id = album.id
#                 added += 1
#             # Set album dates from images
#             imgs = _live(db).filter(DBImage.album_id == album.id).all()
#             timestamps = [i.timestamp for i in imgs if i.timestamp]
#             if timestamps:
#                 album.start_date = min(timestamps)
#                 album.end_date   = max(timestamps)

#         db.commit()
#         logger.info(f"✅ Album created: '{title}' (id={album.id}, {added} images)")
#         return {"status": "created", "id": album.id, "title": album.title, "image_count": added}
#     except Exception as e:
#         db.rollback(); raise HTTPException(500, str(e))
#     finally:
#         db.close()


# @app.delete("/albums/{album_id}/delete")
# def delete_album(album_id: int):
#     """Delete an album (photos are kept, just unlinked from album)."""
#     db = SessionLocal()
#     try:
#         album = db.query(Album).filter(Album.id == album_id).first()
#         if not album:
#             raise HTTPException(404, "Album not found")
#         db.query(DBImage).filter(DBImage.album_id == album_id).update({"album_id": None})
#         db.delete(album); db.commit()
#         return {"status": "deleted", "id": album_id}
#     except HTTPException: raise
#     except Exception as e:
#         db.rollback(); raise HTTPException(500, str(e))
#     finally:
#         db.close()


# @app.delete("/albums/empty/cleanup")
# def cleanup_empty_albums():
#     """Delete all empty manual albums (useful after accidental duplicates)."""
#     db = SessionLocal()
#     try:
#         manual_albums = db.query(Album).filter(Album.type == "manual").all()
#         deleted = []
#         for a in manual_albums:
#             count = db.query(DBImage).filter(DBImage.album_id == a.id).count()
#             if count == 0:
#                 deleted.append(a.title)
#                 db.delete(a)
#         db.commit()
#         return {"status": "done", "deleted": len(deleted), "titles": deleted}
#     except Exception as e:
#         db.rollback(); raise HTTPException(500, str(e))
#     finally:
#         db.close()


# @app.post("/albums/{album_id}/rename")
# def rename_album(album_id: int, title: str = Form(...), description: str = Form("")):
#     """Rename an album and optionally update its description."""
#     db = SessionLocal()
#     try:
#         album = db.query(Album).filter(Album.id == album_id).first()
#         if not album:
#             raise HTTPException(404, "Album not found")
#         album.title = title.strip()
#         if description.strip():
#             album.description = description.strip()
#         db.commit()
#         return {"status": "renamed", "id": album_id, "title": album.title}
#     except HTTPException: raise
#     except Exception as e:
#         db.rollback(); raise HTTPException(500, str(e))
#     finally:
#         db.close()


# @app.post("/albums/{album_id}/add-images")
# def add_images_to_album(album_id: int, image_ids: str = Form(...)):
#     """Add a set of images (comma-separated IDs) to an existing album."""
#     db = SessionLocal()
#     try:
#         album = db.query(Album).filter(Album.id == album_id).first()
#         if not album:
#             raise HTTPException(404, "Album not found")
#         ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
#         updated = _live(db).filter(DBImage.id.in_(ids)).update(
#             {"album_id": album_id}, synchronize_session=False
#         )
#         # Refresh album dates
#         imgs = _live(db).filter(DBImage.album_id == album_id).all()
#         timestamps = [i.timestamp for i in imgs if i.timestamp]
#         if timestamps:
#             album.start_date = min(timestamps)
#             album.end_date   = max(timestamps)
#         db.commit()
#         return {"status": "added", "album_id": album_id, "added": updated}
#     except HTTPException: raise
#     except Exception as e:
#         db.rollback(); raise HTTPException(500, str(e))
#     finally:
#         db.close()


# @app.post("/albums/{album_id}/remove-images")
# def remove_images_from_album(album_id: int, image_ids: str = Form(...)):
#     """Remove images from an album (images are kept, just unlinked)."""
#     db = SessionLocal()
#     try:
#         ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
#         _live(db).filter(DBImage.id.in_(ids), DBImage.album_id == album_id).update(
#             {"album_id": None}, synchronize_session=False
#         )
#         db.commit()
#         return {"status": "removed", "album_id": album_id}
#     except Exception as e:
#         db.rollback(); raise HTTPException(500, str(e))
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # DUPLICATES / EXPLORE / STATS
# # ────────────────────────────────────────────────────────────────────────────
# @app.get("/duplicates")
# def get_duplicates():
#     db = SessionLocal()
#     try:
#         all_images = _live(db).all()
#         if not all_images:
#             return {"status": "found", "duplicate_groups": [], "total_groups": 0}
#         groups    = duplicate_engine.find_duplicates_fast(all_images, hamming_threshold=5)
#         formatted = [{"count": g["count"], "total_size": g["total_size"], "images": [{"id": img["id"], "filename": _img_url(img["filename"]), "thumbnail": _img_url(img["filename"]), "size": img["size"]} for img in g["images"]]} for g in groups]
#         return {"status": "found", "duplicate_groups": formatted, "total_groups": len(formatted)}
#     except Exception as e:
#         logger.error(f"Duplicates error: {e}", exc_info=True)
#         return {"status": "error", "message": str(e), "duplicate_groups": [], "total_groups": 0}
#     finally:
#         db.close()

# @app.get("/explore/random")
# def explore_random(count: int = Query(12)):
#     import random as _random
#     db = SessionLocal()
#     try:
#         all_ids = [row[0] for row in _live(db).with_entities(DBImage.id).all()]
#         if not all_ids:
#             return {"status": "not_found", "results": []}
#         sampled = _random.sample(all_ids, min(count, len(all_ids)))
#         imgs    = db.query(DBImage).filter(DBImage.id.in_(sampled)).all()
#         return {"status": "found", "count": len(imgs), "results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None, "person_count": img.person_count or 0} for img in imgs]}
#     finally:
#         db.close()

# @app.get("/stats")
# def get_stats():
#     import collections
#     db = SessionLocal()
#     try:
#         total_images    = db.query(DBImage).count()
#         total_faces     = db.query(DBFace).count()
#         total_people    = db.query(Person).count()
#         total_albums    = db.query(Album).count()
#         total_favorites = _live(db).filter(DBImage.is_favorite == True).count()
#         total_trashed   = db.query(DBImage).filter(DBImage.is_trashed == True).count()
#         indexed         = search_engine.index.ntotal if search_engine.index else 0
#         # AI object-detection labels (deduplicated per image)
#         ai_counter = collections.Counter()
#         for row in db.query(DBImage.scene_label).filter(DBImage.scene_label != None).all():
#             seen_in_image = set()
#             for tag in (row[0] or "").split(","):
#                 t = tag.strip().lower()
#                 if t and t not in seen_in_image:
#                     ai_counter[t] += 1
#                     seen_in_image.add(t)
#         top_tags = [{"tag": t, "count": c} for t, c in ai_counter.most_common(10)]

#         # User-defined tags
#         user_tag_counter = collections.Counter()
#         for row in db.execute(text("SELECT user_tags FROM images WHERE user_tags IS NOT NULL AND user_tags != '[]' AND (is_trashed IS NULL OR is_trashed=0)")):
#             try:
#                 import json as _j
#                 for t in _j.loads(row[0]): user_tag_counter[t] += 1
#             except Exception: pass
#         top_user_tags = [{"tag": t, "count": c} for t, c in user_tag_counter.most_common(20)]
#         COLOR_BUCKETS = {"red": (220,50,50), "orange": (230,120,40), "yellow": (220,210,50), "green": (50,160,50), "blue": (50,80,220), "purple": (120,50,180), "pink": (230,130,160), "white": (240,240,240), "black": (20,20,20), "gray": (128,128,128), "brown": (140,90,50)}
#         color_dist = {k: 0 for k in COLOR_BUCKETS}
#         for img in _live(db).filter(DBImage.avg_r != None).all():
#             img_rgb = np.array([img.avg_r or 0, img.avg_g or 0, img.avg_b or 0], dtype=np.float32)
#             best_bucket = min(COLOR_BUCKETS, key=lambda k: np.linalg.norm(img_rgb - np.array(COLOR_BUCKETS[k], dtype=np.float32)))
#             color_dist[best_bucket] += 1
#         return {"total_images": total_images, "total_faces": total_faces, "total_people": total_people, "total_albums": total_albums, "total_favorites": total_favorites, "total_trashed": total_trashed, "indexed_vectors": indexed, "top_tags": top_tags, "top_user_tags": top_user_tags, "color_distribution": sorted([{"color": k, "count": v} for k, v in color_dist.items() if v > 0], key=lambda x: x["count"], reverse=True)}
#     finally:
#         db.close()


# # ── Common non-name words returned by BLIP when it doesn't know the name ───
# _BLIP_NON_NAMES = {
#     # articles / pronouns / common words
#     "a", "an", "the", "it", "he", "she", "they", "him", "her", "them",
#     "his", "hers", "their", "we", "our", "you", "your", "my", "i",
#     # question words (critical — BLIP echoes these back)
#     "what", "who", "which", "where", "when", "how", "why", "is", "are",
#     "was", "were", "be", "been", "being", "do", "does", "did", "has",
#     "have", "had", "will", "would", "could", "should", "may", "might",
#     "can", "name", "named", "called", "known",
#     # generic person descriptions
#     "man", "woman", "person", "people", "boy", "girl", "child", "adult",
#     "human", "face", "faces", "subject", "individual", "individuals",
#     "unknown", "someone", "somebody", "anyone", "nobody", "one",
#     # job titles / roles (BLIP often returns these)
#     "celebrity", "actor", "actress", "model", "player", "star", "singer",
#     "politician", "athlete", "musician", "artist", "president", "director",
#     "manager", "leader", "officer", "official", "member", "representative",
#     # visual/image words (BLIP captions)
#     "image", "photo", "picture", "figure", "portrait", "photograph",
#     "this", "that", "these", "those", "here", "there",
#     # caption filler words BLIP uses
#     "main", "shows", "show", "showing", "appears", "appear", "looking",
#     "wearing", "holding", "standing", "sitting", "smiling",
#     "in", "on", "at", "of", "for", "with", "from", "by", "to",
#     "and", "or", "but", "not", "also", "as", "so", "then", "than",
#     "black", "white", "two", "three", "four", "five", "several", "many",
#     # Sports context — appear on jerseys, banners, scoreboards
#     "india", "team", "cup", "match", "captain", "wicket", "wizard",
#     "champion", "cricket", "football", "sport", "league", "series",
#     "trophy", "test", "ipl", "bcci", "odi", "t20", "final", "world",
#     "semi", "quarter", "squad", "player", "coach", "umpire", "referee",
#     # Brand/sponsor words (F1, cricket, general)
#     "petronas", "stake", "marlboro", "ferrari", "mercedes", "mclaren",
#     "alpine", "williams", "haas", "redbull", "birla", "estates", "bank",
#     "mutual", "fund", "limited", "pvt", "company", "corp", "inc", "ltd",
#     "group", "holding", "sponsor", "official", "partner",
#     # Common OCR fragments from Indian sports context
#     "ndia", "iindia", "ldcup", "ineqs", "eironas", "hak", "zon",
#     # Titles / honorifics misread as names
#     "master", "mister", "miss", "mrs", "sir", "dr", "prof", "mr",
#     # Generic English words that look capitalised in headlines
#     "new", "old", "big", "great", "good", "best", "top", "live",
#     "real", "true", "blue", "gold", "king", "queen", "super", "ultra",
# }

# # ── Patterns that definitively mark a word as NOT a person name ─────────────
# def _is_valid_name_word(w: str, allow_short: bool = False) -> bool:
#     """
#     Return True only if `w` looks like part of a real person's name.
#     Rejects: ALL-CAPS brands, too-short initials, non-alpha, known non-names.
#     """
#     w_clean = w.rstrip(".,!?;:'\"")
#     if not w_clean:
#         return False
#     # Must be purely alphabetic (no digits, no hyphens, no underscores)
#     if not w_clean.isalpha():
#         return False
#     # ALL-CAPS words are brand names / acronyms / jersey text, not person names
#     if w_clean == w_clean.upper() and len(w_clean) > 2:
#         return False
#     # Minimum length — real given names are rarely < 3 chars in isolation
#     min_len = 3 if allow_short else 4
#     if len(w_clean) < min_len:
#         return False
#     # Must start with uppercase (proper noun)
#     if not w_clean[0].isupper():
#         return False
#     # Must not be a known non-name
#     if w_clean.lower() in _BLIP_NON_NAMES:
#         return False
#     # Must contain at least one vowel (pure consonant strings = OCR garbage)
#     if not any(c in "aeiouAEIOU" for c in w_clean):
#         return False
#     return True


# def _extract_name_from_vqa(vqa_text: str) -> str:
#     """
#     Parse a BLIP VQA answer to extract a plausible person name.
#     Very strict — only returns something if it looks like a real proper name.
#     Returns empty string if uncertain.
#     """
#     if not vqa_text:
#         return ""
#     text = vqa_text.strip()

#     # Strip common prefixes BLIP adds
#     for prefix in [
#         "the person is ", "this is ", "it is ", "that is ",
#         "his name is ", "her name is ", "the name is ",
#         "the man is ", "the woman is ", "i think it is ",
#         "i think it's ", "i believe it is ", "appears to be ",
#         "looks like ", "this appears to be ", "the answer is ",
#     ]:
#         if text.lower().startswith(prefix):
#             text = text[len(prefix):]

#     # Take just the first 1-2 words
#     words = text.split()[:2]
#     if not words:
#         return ""

#     # Validate each word — allow short (3-char) first names only in 2-word combos
#     if len(words) == 2:
#         w1_ok = _is_valid_name_word(words[0], allow_short=True)
#         w2_ok = _is_valid_name_word(words[1], allow_short=False)
#         if w1_ok and w2_ok:
#             name = f"{words[0].rstrip('.,!?;:')} {words[1].rstrip('.,!?;:')}"
#             return name if 4 <= len(name) <= 35 else ""
#         if w1_ok:
#             w = words[0].rstrip(".,!?;:")
#             return w if 4 <= len(w) <= 35 else ""
#         return ""
#     else:
#         w = words[0].rstrip(".,!?;:")
#         if _is_valid_name_word(w, allow_short=False):
#             return w if 4 <= len(w) <= 35 else ""
#         return ""


# def _auto_name_people(db, person_map: dict):
#     """
#     For each newly created person cluster, try to auto-name them.
#     Strategy (strict):
#       1. VQA "person" answers get highest weight (3 pts each) — most reliable
#       2. OCR proper nouns that appear in MULTIPLE images of the same cluster get 1 pt each
#          (single-image OCR hits are often jersey text / background banners → ignored)
#       3. Only assign a name if the top candidate has >= 50% of photos voting for it
#          AND if it's a VQA-sourced name, OR >= 2 images (to prevent single-image OCR garbage)
#     """
#     import json as _jj
#     from collections import Counter

#     for label, person_id in person_map.items():
#         person = db.query(Person).filter(Person.id == person_id).first()
#         if not person:
#             continue
#         # Only auto-name if still using the default label
#         if not person.name.startswith("Person "):
#             continue

#         face_records = db.query(DBFace).filter(DBFace.person_id == person_id).all()
#         img_ids = list({f.image_id for f in face_records if f.image_id})
#         if not img_ids:
#             continue

#         images = db.query(DBImage).filter(DBImage.id.in_(img_ids)).all()
#         n_images = len(images)

#         # Separate counters: VQA (trusted) vs OCR (needs cross-image confirmation)
#         vqa_counter = Counter()
#         ocr_counter = Counter()   # counts how many IMAGES contain each OCR name

#         for img in images:
#             # ── VQA: most reliable, use directly ────────────────────────────
#             if img.caption_vqa:
#                 try:
#                     vqa = _jj.loads(img.caption_vqa)
#                     vqa_name = _extract_name_from_vqa(vqa.get("person", ""))
#                     if vqa_name:
#                         vqa_counter[vqa_name] += 1
#                 except Exception:
#                     pass

#             # ── OCR: scan for proper-noun pairs and single words ─────────────
#             # Only count a name ONCE per image (to prevent a single jersey from
#             # winning just because the text appears 10 times in one photo).
#             ocr = img.ocr_text_enhanced or ""
#             if ocr:
#                 seen_in_this_image = set()
#                 words = ocr.split()
#                 for i, w in enumerate(words):
#                     # Try two-word name first
#                     if i + 1 < len(words):
#                         w2 = words[i + 1].rstrip(".,!?;:'\"")
#                         w1 = w.rstrip(".,!?;:'\"")
#                         if (_is_valid_name_word(w1, allow_short=True) and
#                                 _is_valid_name_word(w2, allow_short=False)):
#                             two_word = f"{w1} {w2}"
#                             if two_word not in seen_in_this_image:
#                                 seen_in_this_image.add(two_word)
#                                 ocr_counter[two_word] += 1

#                     # Single word fallback
#                     w1 = w.rstrip(".,!?;:'\"")
#                     if (_is_valid_name_word(w1, allow_short=False) and
#                             w1 not in seen_in_this_image):
#                         seen_in_this_image.add(w1)
#                         ocr_counter[w1] += 1

#         # ── Decision logic ────────────────────────────────────────────────────
#         best_name = None
#         best_score = 0

#         # VQA wins if it appears in >= 25% of images (e.g. 1 out of 4)
#         if vqa_counter:
#             top_vqa, top_vqa_count = vqa_counter.most_common(1)[0]
#             min_vqa = max(1, n_images * 0.25)
#             if top_vqa_count >= min_vqa:
#                 best_name  = top_vqa
#                 best_score = top_vqa_count * 3  # weighted

#         # OCR only wins if:
#         # (a) NO VQA name found, AND
#         # (b) name appears in >= 2 different images (cross-image confirmation), AND
#         # (c) appears in >= 40% of the cluster's images
#         if not best_name and ocr_counter:
#             top_ocr, top_ocr_count = ocr_counter.most_common(1)[0]
#             min_ocr_images = max(2, n_images * 0.40)
#             if top_ocr_count >= min_ocr_images:
#                 best_name  = top_ocr
#                 best_score = top_ocr_count

#         if best_name:
#             person.name = best_name
#             logger.info(f"🏷️  Auto-named Person {person_id} → '{best_name}' "
#                         f"(score={best_score}, vqa={dict(vqa_counter.most_common(3))}, "
#                         f"n_images={n_images})")
#         else:
#             top_vqa_str = str(vqa_counter.most_common(1)) if vqa_counter else "none"
#             top_ocr_str = str(ocr_counter.most_common(3)) if ocr_counter else "none"
#             logger.info(f"⚠️  No confident name for Person {person_id} "
#                         f"(n_images={n_images}, vqa={top_vqa_str}, top_ocr={top_ocr_str})")


# # ────────────────────────────────────────────────────────────────────────────
# # RECLUSTER
# # ────────────────────────────────────────────────────────────────────────────
# @app.post("/recluster")

# def recluster():
#     db = SessionLocal()
#     try:
#         logger.info("🔄 Recluster: saving manually assigned names before reset…")

#         # ── Save manually named people BEFORE wiping everything ─────────────
#         # For each person with a non-default name, remember which face embeddings
#         # (as indices into the face table) belonged to them.
#         saved_names = {}  # face_id_set -> name
#         for person in db.query(Person).all():
#             pname = (person.name or "").strip()
#             # Only keep explicitly renamed people (skip "Person N" defaults)
#             if re.match(r"^person\s+\d+$", pname.lower()):
#                 continue
#             if not pname or pname.lower() in ("unknown", ""):
#                 continue
#             face_ids = frozenset(
#                 f.id for f in db.query(DBFace).filter(DBFace.person_id == person.id).all()
#             )
#             if face_ids:
#                 saved_names[face_ids] = pname
#                 logger.info(f"💾 Saved name '{pname}' ({len(face_ids)} faces)")

#         logger.info(f"🔄 Recluster: clearing old assignments…")
#         db.query(DBFace).update({"person_id": None})
#         db.query(Person).delete()

#         # ── CRITICAL: only clear album_id for EVENT album images ─────────────
#         # db.query(DBImage).update({"album_id": None})  ← old code wiped MANUAL albums too
#         # Manual albums (type="manual") must be preserved — user curated them.
#         # Only event albums (type="event") are rebuilt automatically.
#         event_album_ids = [
#             a.id for a in db.query(Album).filter(Album.type == "event").all()
#         ]
#         if event_album_ids:
#             db.query(DBImage).filter(
#                 DBImage.album_id.in_(event_album_ids)
#             ).update({"album_id": None}, synchronize_session=False)

#         db.query(Album).filter(Album.type == "event").delete()
#         db.commit()

#         # ── Load face records with image dimensions for size-filtering ────────
#         # Background faces (tiny, far-away people) corrupt clusters — the same
#         # main person ends up split into multiple clusters because their face
#         # sometimes appears blurry/small in group shots.
#         # Strategy: for each image, only cluster faces that are "prominent":
#         #   - face area >= MIN_FACE_RATIO of image area (ignores tiny bystanders)
#         #   - if multiple faces pass the threshold, keep top MAX_FACES_PER_IMAGE
#         #     by size (supports group photos with 2-3 main subjects)
#         MIN_FACE_RATIO   = 0.005   # face must cover >= 0.5% of image area
#                                     # 1920×1080 image → face must be >= ~100×100 px
#         MAX_FACES_PER_IMAGE = 4    # keep at most 4 largest faces per image

#         # Build image dimension lookup: image_id → (width, height)
#         img_dims = {}
#         for img in db.query(DBImage.id, DBImage.width, DBImage.height).all():
#             if img.width and img.height:
#                 img_dims[img.id] = (img.width, img.height)

#         face_records            = db.query(DBFace).filter(DBFace.face_embedding != None).all()
#         embeddings, valid_faces = [], []
#         skipped = 0
#         bg_filtered = 0

#         # Group faces by image to allow per-image top-N selection
#         from collections import defaultdict
#         import json as _jfilt
#         faces_by_image = defaultdict(list)
#         for fr in face_records:
#             if not fr.image_id:
#                 continue
#             try:
#                 emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
#                 if emb.shape[0] != 512:
#                     skipped += 1
#                     continue
#                 # Parse bbox to get face area
#                 face_area_ratio = 1.0  # default: include if no bbox/dims available
#                 if fr.bbox and fr.image_id in img_dims:
#                     try:
#                         bbox = _jfilt.loads(fr.bbox)  # [x1, y1, x2, y2]
#                         x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
#                         face_area = max(0, x2 - x1) * max(0, y2 - y1)
#                         img_w, img_h = img_dims[fr.image_id]
#                         img_area = img_w * img_h
#                         face_area_ratio = face_area / img_area if img_area > 0 else 1.0
#                     except Exception:
#                         face_area_ratio = 1.0  # parse failed → include by default
#                 faces_by_image[fr.image_id].append((face_area_ratio, fr, emb))
#             except Exception as e:
#                 logger.warning(f"Bad embedding face {fr.id}: {e}")
#                 skipped += 1

#         # For each image: filter by MIN_FACE_RATIO, then keep top MAX_FACES_PER_IMAGE
#         for image_id, face_list in faces_by_image.items():
#             # Sort by area ratio descending (largest face first)
#             face_list.sort(key=lambda x: x[0], reverse=True)

#             # Apply area filter — but always keep at least 1 face per image
#             # (portrait shots where the face fills the frame pass trivially)
#             passed = [item for item in face_list if item[0] >= MIN_FACE_RATIO]
#             if not passed:
#                 # All faces were small — keep the single largest one anyway
#                 # (could be a small portrait photo shot from far away)
#                 passed = face_list[:1]
#                 logger.debug(f"🔍 Image {image_id}: all faces small, keeping largest "
#                              f"(ratio={face_list[0][0]:.4f})")

#             # Cap at MAX_FACES_PER_IMAGE
#             kept = passed[:MAX_FACES_PER_IMAGE]
#             dropped = len(face_list) - len(kept)
#             if dropped > 0:
#                 bg_filtered += dropped
#                 logger.debug(f"🔍 Image {image_id}: kept {len(kept)}/{len(face_list)} faces "
#                              f"(dropped {dropped} background faces, "
#                              f"min_ratio={kept[-1][0]:.4f})")

#             for _, fr, emb in kept:
#                 embeddings.append(emb)
#                 valid_faces.append(fr)

#         logger.info(f"👥 {len(embeddings)} prominent face embeddings "
#                     f"({skipped} invalid, {bg_filtered} background faces filtered out)")
#         people_count = 0
#         if embeddings:
#             labels     = face_engine.cluster_faces(embeddings)
#             person_map = {}
#             for i, label in enumerate(labels):
#                 if label == -1: continue
#                 if label not in person_map:
#                     p = Person(name=f"Person {label + 1}")
#                     db.add(p); db.flush()
#                     person_map[label] = p.id; people_count += 1
#                 valid_faces[i].person_id = person_map[label]
#             db.commit()
#             logger.info(f"✅ {people_count} people created")

#             # ── Restore manually assigned names to matching new clusters ────
#             if saved_names:
#                 restored = 0
#                 # Build map: face_id -> new person_id
#                 face_to_person = {}
#                 for face in db.query(DBFace).filter(DBFace.person_id != None).all():
#                     face_to_person[face.id] = face.person_id

#                 for old_face_ids, saved_name in saved_names.items():
#                     # Find which new person cluster has the most overlap with old faces
#                     overlap_count = {}
#                     for fid in old_face_ids:
#                         pid = face_to_person.get(fid)
#                         if pid:
#                             overlap_count[pid] = overlap_count.get(pid, 0) + 1

#                     if not overlap_count:
#                         continue
#                     # Best matching cluster
#                     best_pid = max(overlap_count, key=overlap_count.get)
#                     best_overlap = overlap_count[best_pid]
#                     # Only restore if at least 30% of faces match
#                     min_match = max(1, len(old_face_ids) * 0.30)
#                     if best_overlap >= min_match:
#                         person = db.query(Person).filter(Person.id == best_pid).first()
#                         if person and re.match(r"^person\s+\d+$", (person.name or "").lower()):
#                             person.name = saved_name
#                             restored += 1
#                             logger.info(f"✅ Restored name '{saved_name}' → Person {best_pid} ({best_overlap}/{len(old_face_ids)} faces matched)")
#                 db.commit()
#                 logger.info(f"🏷️  Restored {restored}/{len(saved_names)} manually assigned names")

#             # Auto-name remaining unnamed clusters from BLIP captions / VQA
#             _auto_name_people(db, person_map)
#             db.commit()
#         all_images   = db.query(DBImage).all()
#         albums_count = 0
#         if all_images:
#             metadata = [{"id": img.id, "lat": img.lat or 0.0, "lon": img.lon or 0.0, "timestamp": img.timestamp} for img in all_images if img.timestamp]
#             if metadata:
#                 album_labels = clustering_engine.detect_events(metadata)
#                 album_map    = {}
#                 for i, label in enumerate(album_labels):
#                     if label == -1: continue
#                     if label not in album_map:
#                         cluster_meta = [metadata[j] for j, l in enumerate(album_labels) if l == label]
#                         ts_list      = [m["timestamp"] for m in cluster_meta if m["timestamp"]]
#                         start_d      = min(ts_list) if ts_list else None
#                         end_d        = max(ts_list) if ts_list else None
#                         if start_d:
#                             if end_d and end_d.date() != start_d.date():
#                                 # Multi-day: "Mar 4 – 7, 2026"
#                                 if end_d.month == start_d.month:
#                                     title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%d, %Y')}"
#                                 else:
#                                     title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%b %d, %Y')}"
#                             else:
#                                 # Single day: "Mar 4, 2026"
#                                 title = start_d.strftime("%b %d, %Y")
#                         else:
#                             title = f"Event {label + 1}"
#                         new_album = Album(title=title, type="event", start_date=start_d, end_date=end_d)
#                         db.add(new_album); db.flush()
#                         album_map[label] = new_album.id; albums_count += 1
#                     db.query(DBImage).filter(DBImage.id == metadata[i]["id"]).update({"album_id": album_map[label]})
#                 db.commit()
#                 logger.info(f"✅ {albums_count} albums created")
#         return {"status": "done", "people": people_count, "albums": albums_count}
#     except Exception as e:
#         db.rollback()
#         logger.error(f"❌ Recluster failed: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # UPLOAD
# # ────────────────────────────────────────────────────────────────────────────
# @app.post("/upload")
# async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
#     ext = os.path.splitext(file.filename)[1].lower()
#     if ext not in [".jpg", ".jpeg", ".png"]:
#         raise HTTPException(status_code=400, detail="Only JPG and PNG supported.")
#     filename  = f"{uuid.uuid4()}{ext}"
#     file_path = os.path.join(IMAGE_DIR, filename)
#     db        = SessionLocal()
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         from PIL import Image as PILImage
#         width = height = None
#         avg_r = avg_g = avg_b = 0.0
#         try:
#             img_pil       = PILImage.open(file_path).convert("RGB")
#             width, height = img_pil.size
#             arr = np.array(img_pil)
#             # Use dominant color (most common cluster) instead of plain average
#             # Resize to 64x64 for speed, then find the peak of the color histogram
#             import colorsys as _cs
#             small = np.array(img_pil.resize((64, 64)), dtype=np.float32)
#             r_vals = small[:,:,0].flatten()
#             g_vals = small[:,:,1].flatten()
#             b_vals = small[:,:,2].flatten()
#             # Weight pixels by their saturation (ignore grey/white/black background)
#             weights = []
#             for ri, gi, bi in zip(r_vals, g_vals, b_vals):
#                 _, s, v = _cs.rgb_to_hsv(ri/255, gi/255, bi/255)
#                 weights.append(s * v + 0.01)  # saturated bright pixels count more
#             weights = np.array(weights)
#             total_w = weights.sum()
#             avg_r = float(np.dot(r_vals, weights) / total_w)
#             avg_g = float(np.dot(g_vals, weights) / total_w)
#             avg_b = float(np.dot(b_vals, weights) / total_w)
#         except Exception:
#             pass
#         clip_emb = None
#         try:
#             clip_emb = search_engine.get_image_embedding(file_path)
#         except Exception:
#             pass
#         scene_label  = ""
#         person_count = 0
#         try:
#             objects      = detector_engine.detect_objects(file_path, threshold=0.5)
#             scene_label  = ", ".join(objects) if objects else ""
#             person_count = detector_engine.detect_persons(file_path)
#         except Exception:
#             pass
#         img_record = DBImage(filename=filename, original_path=file_path, timestamp=datetime_module.datetime.now(), width=width, height=height, avg_r=avg_r, avg_g=avg_g, avg_b=avg_b, scene_label=scene_label, person_count=person_count, ocr_text_enhanced="", ocr_keywords="[]", ocr_confidence=0.0, detected_language="en", caption_short="", caption_detailed="", caption_vqa="{}", quality_score=0.0, quality_level="Processing...", sharpness=0.0, exposure=0.0, contrast=0.0, composition=0.0, emotion_data="[]", dominant_emotion="neutral", face_emotion_count=0, aesthetic_score=0.0, aesthetic_rating="Processing...")
#         db.add(img_record); db.flush()
#         if clip_emb is not None:
#             try:
#                 if search_engine.index is None:
#                     search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
#                 new_vec = clip_emb.reshape(1, -1).astype("float32")
#                 faiss.normalize_L2(new_vec)
#                 search_engine.index.add_with_ids(new_vec, np.array([img_record.id], dtype="int64"))
#                 faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
#             except Exception as e:
#                 logger.warning(f"FAISS update failed: {e}")
#         face_count = 0
#         try:
#             faces = face_engine.detect_faces(file_path)
#             if faces and width and height:
#                 img_area = width * height
#                 # Sort faces by area descending (largest = most prominent first)
#                 def _face_area(f):
#                     b = f["bbox"]  # [x1, y1, x2, y2]
#                     return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
#                 faces_sorted = sorted(faces, key=_face_area, reverse=True)
#                 # Keep only prominent faces (>= 0.5% of image area), max 4
#                 prominent = [f for f in faces_sorted
#                              if _face_area(f) / img_area >= 0.005]
#                 if not prominent:
#                     prominent = faces_sorted[:1]  # always keep largest face
#                 faces = prominent[:4]
#             for face in faces:
#                 emb = face["embedding"].astype(np.float32)
#                 db.add(DBFace(image_id=img_record.id, bbox=_json.dumps(face["bbox"]), face_embedding=emb.tobytes()))
#                 face_count += 1
#         except Exception as e:
#             logger.warning(f"Face detection failed: {e}")
#         db.commit()
#         image_id = img_record.id
#         if background_tasks is not None:
#             background_tasks.add_task(_enrich_image, image_id, file_path)
#             should_trigger_recluster(background_tasks)
#         else:
#             _enrich_image(image_id, file_path)
#         logger.info(f"✅ Upload done: {filename} (id={image_id})")
#         return {"status": "success", "id": image_id, "filename": filename, "person_count": person_count, "face_count": face_count, "quality_score": 0.0, "quality_level": "Processing...", "caption": "", "dominant_emotion": "neutral", "emotion_count": 0, "aesthetic_score": 0.0, "note": "ML enrichment running in background"}
#     except Exception as e:
#         db.rollback()
#         if os.path.exists(file_path): os.remove(file_path)
#         logger.error(f"Upload failed: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # REPROCESS
# # ────────────────────────────────────────────────────────────────────────────
# @app.post("/reprocess_images")
# async def reprocess_images():
#     db = SessionLocal()
#     processed, failed = 0, 0
#     try:
#         images = db.query(DBImage).filter((DBImage.caption_short == None) | (DBImage.caption_short == "") | (DBImage.quality_level == "Processing...")).all()
#         if not images:
#             return {"success": True, "processed": 0, "failed": 0}
#         for img in images:
#             try:
#                 if not img.original_path or not os.path.exists(img.original_path):
#                     failed += 1; continue
#                 _enrich_image(img.id, img.original_path); processed += 1
#             except Exception:
#                 failed += 1
#         return {"success": True, "processed": processed, "failed": failed}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         db.close()


# # ────────────────────────────────────────────────────────────────────────────
# # RECAPTION — re-run BLIP on images with missing or poor captions
# # ────────────────────────────────────────────────────────────────────────────
# @app.post("/recaption")
# async def recaption_images(background_tasks: BackgroundTasks, force_all: bool = False):
#     """
#     Re-run BLIP captioning on images that:
#     - Have no caption (caption_short is NULL or empty)
#     - Have a placeholder/error caption
#     - force_all=True: re-caption every image (useful after switching BLIP model)

#     Runs in the background so the API returns immediately.
#     Progress can be checked via GET /recaption/status
#     """
#     db = SessionLocal()
#     try:
#         if force_all:
#             images = _live(db).all()
#         else:
#             # Only images missing captions
#             images = _live(db).filter(
#                 (DBImage.caption_short == None) |
#                 (DBImage.caption_short == "") |
#                 (DBImage.caption_short == "Processing...")
#             ).all()

#         pairs = [
#             (img.id, img.original_path or os.path.join(IMAGE_DIR, img.filename))
#             for img in images
#         ]
#         # Filter to files that actually exist
#         pairs = [(iid, fp) for iid, fp in pairs if os.path.exists(fp)]
#     finally:
#         db.close()

#     if not pairs:
#         return {"status": "nothing_to_do", "count": 0,
#                 "message": "All images already have captions. Use force_all=true to re-caption everything."}

#     def _run_recaption(image_pairs):
#         done = 0
#         for img_id, fpath in image_pairs:
#             try:
#                 db2 = SessionLocal()
#                 try:
#                     caption_short    = captioning_engine.generate_caption(fpath, max_length=20)
#                     caption_detailed = captioning_engine.generate_caption(fpath, max_length=60)
#                     vqa_subject = captioning_engine.answer_visual_question(
#                         fpath, "What is the main subject in this image?"
#                     )
#                     # Check if this image has people for VQA person question
#                     img_chk = db2.query(DBImage).filter(DBImage.id == img_id).first()
#                     vqa_person = ""
#                     if img_chk and (img_chk.person_count or 0) > 0:
#                         vqa_person = captioning_engine.answer_visual_question(fpath, "who is this?")
#                         if vqa_person and vqa_person.lower().strip() in {
#                             "man","woman","person","a man","a woman","a person","boy","girl",
#                             "child","human","unknown","celebrity","actor","actress","no","yes",
#                             "none","i don't know",
#                         }:
#                             vqa_person = ""

#                     db2.query(DBImage).filter(DBImage.id == img_id).update({
#                         "caption_short":    caption_short,
#                         "caption_detailed": caption_detailed,
#                         "caption_vqa":      _json.dumps({
#                             "subject": vqa_subject,
#                             "person":  vqa_person,
#                         } if (vqa_subject or vqa_person) else {}),
#                         "caption_timestamp": datetime_module.datetime.now(),
#                     }, synchronize_session=False)
#                     db2.commit()
#                     done += 1
#                     logger.info(f"✅ Recaptioned [{img_id}]: {caption_short}")
#                 finally:
#                     db2.close()
#             except Exception as e:
#                 logger.warning(f"⚠️ Recaption failed [{img_id}]: {e}")
#         logger.info(f"✅ Recaption complete: {done}/{len(image_pairs)} images")

#     background_tasks.add_task(_run_recaption, pairs)
#     return {
#         "status": "started",
#         "count": len(pairs),
#         "message": f"Re-captioning {len(pairs)} images in background. "
#                    f"{'(all images)' if force_all else '(missing captions only)'}",
#     }


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
import json as _json
from enhanced_ocr_engine import ocr_engine
from image_captioning_engine import captioning_engine
from quality_emotion_aesthetic_engines import (
    image_quality,
    emotion_detection,
    aesthetic_scoring
)

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
from fastapi.responses import FileResponse
from sqlalchemy import text
import re
import json
from api_endpoints import router as dl_router
from voice_route import router as voice_router
from features_router import router as feat_router, ensure_extra_columns   # ← only import here

_BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR        = os.path.normpath(os.path.join(_BASE_DIR, "..", "data", "images"))
FAISS_INDEX_PATH = os.path.normpath(os.path.join(_BASE_DIR, "..", "data", "index.faiss"))

FACE_MATCH_THRESHOLD  = float(os.environ.get("FACE_MATCH_THRESHOLD",  0.75))
FACE_MATCH_NEIGHBORS  = int(os.environ.get("FACE_MATCH_NEIGHBORS",    5))
FACE_MATCH_VOTE_RATIO = float(os.environ.get("FACE_MATCH_VOTE_RATIO", 0.6))
RECLUSTER_ON_UPLOAD   = os.environ.get("RECLUSTER_ON_UPLOAD", "true").lower() in ("1", "true", "yes")
RECLUSTER_BATCH_SIZE  = int(os.environ.get("RECLUSTER_BATCH_SIZE", 10))

CLIP_SCORE_MIN  = float(os.environ.get("CLIP_SCORE_MIN",  0.14))  # 0.14: catches football/soccer/cat/dog matches
THRESHOLD_FLOOR = float(os.environ.get("THRESHOLD_FLOOR", 0.22))
ADAPTIVE_RATIO  = float(os.environ.get("ADAPTIVE_RATIO",  0.92))
FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.13))  # 0.13: matches CLIP threshold

RECLUSTER_COUNTER_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "recluster_counter.txt")
RECLUSTER_TIMER_SECONDS = float(os.environ.get("RECLUSTER_TIMER_SECONDS", 8.0))  # 8s: fire soon after upload
recluster_last_triggered = None

COLOR_SCORE_MAP = {
    'red':    (1.0, 0,    0),    'blue':   (0,   0,    1.0),
    'green':  (0,   1.0,  0),    'yellow': (1.0, 1.0,  0),
    'orange': (1.0, 0.5,  0),    'purple': (0.5, 0,    0.5),
    'pink':   (1.0, 0.75, 0.8),  'black':  (0,   0,    0),
    'white':  (1,   1,    1),    'gray':   (0.5, 0.5,  0.5),
    'brown':  (0.6, 0.4,  0.2),
}


# ── Emotion emojis → search terms (intercept before CLIP) ───────────────────
EMOTION_EMOJI_MAP = {
    "😊": "happy",  "😄": "happy",  "😁": "happy",  "🙂": "happy",  "😀": "happy",
    "😃": "happy",  "🤗": "happy",  "😆": "happy",  "😂": "happy",  "🥰": "happy",
    "😍": "happy",  "😎": "happy",  "🥳": "happy",  "😇": "happy",
    "😢": "sad",    "😭": "sad",    "😔": "sad",    "😟": "sad",    "🥺": "sad",
    "😞": "sad",    "😿": "sad",    "💔": "sad",
    "😠": "angry",  "😡": "angry",  "🤬": "angry",  "😤": "angry",  "👿": "angry",
    "😲": "surprised", "😮": "surprised", "🤯": "surprised", "😱": "surprised",
    "🤢": "disgusted", "🤮": "disgusted", "😷": "disgusted",
    "😨": "fearful",   "😰": "fearful",   "😱": "fearful",   "😧": "fearful",
    "😐": "neutral",   "😑": "neutral",   "🙄": "neutral",   "😶": "neutral",
}

# Extended scene emoji map (supplements search_engine.py EMOJI_MAP)
SCENE_EMOJI_MAP = {
    "🌅": "sunset golden hour", "🌃": "city night",   "🏕️": "camping outdoor",
    "🌴": "tropical palm tree", "⛰️": "mountain",     "🌊": "ocean waves beach",
    "🎊": "party celebration",  "👪": "family group",  "🤳": "selfie portrait",
    "🐕": "dog",                "🐈": "cat",            "🦁": "lion animal",
    "🍽️": "food meal",          "☕": "coffee",         "🍰": "cake dessert",
    "🏋️": "gym exercise",       "🧘": "yoga meditation","🏊": "swimming pool",
    "🌸": "flowers spring",     "🍂": "autumn fall leaves",
    "🎸": "music guitar",       "📸": "camera photography",
    "⚽": "soccer football sport", "🏀": "basketball sport",
    "🏈": "american football sport", "🎾": "tennis sport",
    "🏏": "cricket sport bat", "🏒": "hockey sport",
    "⚾": "baseball sport",    "🥊": "boxing sport",
}

# Sport/activity keyword expansions (applied in _clean_query)
SPORT_SYNONYMS = {
    "football": "football soccer sport player",
    "soccer":   "soccer football sport player",
    "cricket":  "cricket sport bat player",
    "basketball": "basketball sport player",
    "tennis":   "tennis sport player",
    "swimming": "swimming pool sport",
    "running":  "running jogging sport",
    "cycling":  "cycling bicycle sport",
    "gym":      "gym workout exercise",
    "yoga":     "yoga meditation exercise",
}

# ── Text keyword → emotion mapping ─────────────────────────────────────────
EMOTION_KEYWORDS = {
    "happy":     ["happy", "happiness", "smiling", "smile", "laughing", "laugh",
                  "joy", "joyful", "cheerful", "glad", "delighted", "pleased"],
    "sad":       ["sad", "sadness", "crying", "cry", "unhappy", "upset",
                  "depressed", "gloomy", "sorrow", "sorrowful", "tears"],
    "angry":     ["angry", "anger", "mad", "furious", "rage", "annoyed", "irritated"],
    "surprised": ["surprised", "surprise", "shocked", "shock", "amazed", "astonished"],
    "fearful":   ["fearful", "fear", "scared", "afraid", "frightened", "terrified"],
    "disgusted": ["disgusted", "disgust", "disgusting"],
    "neutral":   ["neutral", "expressionless", "blank"],
}

def _extract_emotion_from_query(query: str):
    """Return emotion name if query contains an emotion emoji or keyword, else None."""
    # Check emojis first
    for emoji, emo in EMOTION_EMOJI_MAP.items():
        if emoji in query:
            return emo
    # Check text keywords
    q_lower = query.lower()
    for emo, keywords in EMOTION_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return emo
    return None

def _expand_query_emojis(query: str) -> str:
    """Replace scene emojis with text equivalents for better CLIP embedding."""
    result = query
    for emoji, text in SCENE_EMOJI_MAP.items():
        result = result.replace(emoji, " " + text + " ")
    return " ".join(result.split())


def _live(db):
    return db.query(DBImage).filter(
        (DBImage.is_trashed == False) | (DBImage.is_trashed == None)
    )


def _img_url(filename: str) -> str:
    if not filename:
        return None
    return os.path.basename(filename)


def _user_tags(img) -> list:
    """Safely decode the user_tags JSON column into a Python list."""
    try:
        v = getattr(img, 'user_tags', None)
        return json.loads(v) if v and v not in ('[]', '') else []
    except Exception:
        return []


def _clean_query(query: str) -> str:
    import unicodedata, re as _re
    processed = resolve_query(query)
    result = []
    for char in processed:
        cp = ord(char)
        if cp > 127:
            cat = unicodedata.category(char)
            if cat in ("So", "Sm", "Sk", "Mn"):
                try:
                    name = unicodedata.name(char, "").lower()
                    name = name.replace(" sign", "").replace(" symbol", "")
                    name = _re.sub(r"\bwith\b.*", "", name).strip()
                    result.append(" " + name + " ")
                except Exception:
                    pass
            else:
                result.append(char)
        else:
            result.append(char)
    cleaned = _re.sub(r"\s+", " ", "".join(result)).strip()
    return cleaned if cleaned else processed


def _startup_fix_filenames():
    db = SessionLocal()
    try:
        fixed = 0
        rows = db.query(DBImage).filter(DBImage.filename.contains("/")).all()
        for r in rows:
            r.filename = os.path.basename(r.filename)
            fixed += 1
        if fixed:
            db.commit()
            logger.info(f"Startup fix: normalised {fixed} filenames in DB")
    except Exception as e:
        db.rollback()
        logger.warning(f"Startup filename fix failed: {e}")
    finally:
        db.close()


def _enrich_image(image_id: int, file_path: str):
    db = SessionLocal()
    try:
        img_record = db.query(DBImage).filter(DBImage.id == image_id).first()
        if not img_record:
            return
        try:
            img_record.caption_short    = captioning_engine.generate_caption(file_path, max_length=20)
            img_record.caption_detailed = captioning_engine.generate_caption(file_path, max_length=60)
            vqa_subject = captioning_engine.answer_visual_question(file_path, "What is the main subject in this image?")
            vqa_person  = ""
            if img_record.person_count and img_record.person_count > 0:
                # Ask BLIP to describe the person — identity recognition
                # works best with very short questions
                vqa_person = captioning_engine.answer_visual_question(
                    file_path, "who is this?"
                )
                # Discard generic answers immediately
                if vqa_person and vqa_person.lower().strip() in {
                    "man", "woman", "person", "a man", "a woman", "a person",
                    "boy", "girl", "child", "human", "unknown", "celebrity",
                    "actor", "actress", "no", "yes", "none", "i don't know",
                }:
                    vqa_person = ""
            img_record.caption_vqa = _json.dumps({
                "subject": vqa_subject,
                "person":  vqa_person,
            } if (vqa_subject or vqa_person) else {})
            img_record.caption_timestamp = datetime_module.datetime.now()
            logger.info(f"✅ Caption [{image_id}]: {img_record.caption_short}")
        except Exception as e:
            logger.warning(f"⚠️  Caption failed [{image_id}]: {e}")
        try:
            img_record.ocr_text_enhanced = ocr_engine.extract_text(file_path)
            kw = ocr_engine.extract_text_with_confidence(file_path)
            img_record.ocr_keywords   = _json.dumps([i["text"] for i in kw if i["confidence"] > 0.7])
            img_record.ocr_confidence = float(np.mean([i["confidence"] for i in kw])) if kw else 0.0
            logger.info(f"✅ OCR [{image_id}]")
        except Exception as e:
            logger.warning(f"⚠️  OCR failed [{image_id}]: {e}")
        try:
            q = image_quality.assess_overall_quality(file_path)
            img_record.quality_score = q.get("overall", 0)
            img_record.quality_level = q.get("quality_level", "Unknown")
            img_record.sharpness     = q.get("sharpness", 0)
            img_record.exposure      = q.get("exposure", 0)
            img_record.contrast      = q.get("contrast", 0)
            img_record.composition   = q.get("composition", 0)
            logger.info(f"✅ Quality [{image_id}]: {img_record.quality_level}")
        except Exception as e:
            logger.warning(f"⚠️  Quality failed [{image_id}]: {e}")
        try:
            # ── GATE: only run emotion model if InsightFace detected real human faces ──
            # InsightFace (ArcFace) is accurate at human faces — won't detect cat faces,
            # car headlights, or turtle heads. OpenCV Haar cascade (used by the emotion
            # engine) will detect ALL of these as "faces" and classify them wrongly.
            # Using the faces table count ensures we only run on real human images.
            _face_count_check = db.query(DBFace).filter(DBFace.image_id == image_id).count()
            if _face_count_check == 0:
                # No human faces detected by InsightFace → skip emotion model entirely
                img_record.dominant_emotion   = "neutral"
                img_record.face_emotion_count = 0
                img_record.emotion_data       = "[]"
                logger.info(f"⏭️  Emotion [{image_id}]: skipped (no InsightFace faces)")
            else:
                ed = emotion_detection.detect_emotions(file_path)
                # ── Confidence filter: only keep high-confidence detections ──────────
                # Low-confidence results (< 0.50) from Haar cascade on non-frontal or
                # partially visible faces produce random emotion labels. Filter them out.
                ed = [e for e in ed if e.get("confidence", 0) >= 0.50]
                img_record.face_emotion_count = len(ed)
                img_record.dominant_emotion   = ed[0]["emotion"] if ed else "neutral"
                img_record.emotion_data       = _json.dumps(ed)
                logger.info(f"✅ Emotion [{image_id}]: {img_record.dominant_emotion} ({len(ed)} faces)")
        except Exception as e:
            logger.warning(f"⚠️  Emotion failed [{image_id}]: {e}")
        try:
            a = aesthetic_scoring.score_aesthetics(file_path)
            img_record.aesthetic_score  = a.get("aesthetic_score", 0)
            img_record.aesthetic_rating = a.get("rating", "Unknown")
            logger.info(f"✅ Aesthetic [{image_id}]: {img_record.aesthetic_rating}")
        except Exception as e:
            logger.warning(f"⚠️  Aesthetic failed [{image_id}]: {e}")
        db.commit()
        logger.info(f"✅ Enrichment done for image {image_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Enrichment failed [{image_id}]: {e}", exc_info=True)
    finally:
        db.close()


def should_trigger_recluster(background_tasks):
    global recluster_last_triggered
    if not background_tasks:
        return
    try:
        import threading as _threading
        now = datetime_module.datetime.now()
        # Rate-limit: don't fire again if recluster started within 15 seconds.
        if recluster_last_triggered:
            elapsed = (now - recluster_last_triggered).total_seconds()
            if elapsed < 15:
                logger.debug(f"Recluster skipped — last ran {elapsed:.0f}s ago")
                return
        t = _threading.Thread(target=recluster, daemon=True)
        t.start()
        recluster_last_triggered = now
        logger.info("📊 Recluster triggered (background thread)")
    except Exception as e:
        logger.warning(f"Recluster trigger failed: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_extra_columns()          # ← runs safely here, after DB is initialised
    _startup_fix_filenames()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            search_engine.index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"✅ Index loaded ({search_engine.index.ntotal} vectors)")
        except Exception as e:
            logger.error(f"Index load failed: {e}")
            search_engine.index = None
    try:
        _db = SessionLocal()
        try:
            faces_total    = _db.query(DBFace).filter(DBFace.face_embedding != None).count()
            faces_assigned = _db.query(DBFace).filter(DBFace.face_embedding != None, DBFace.person_id != None).count()
            people_total  = _db.query(Person).count()
            imgs_in_album = _db.query(DBImage).filter(DBImage.album_id != None).count()
            total_imgs    = _db.query(DBImage).count()
            needs_face_repair  = faces_total > 0 and (faces_assigned == 0 or people_total == 0)
            needs_album_repair = total_imgs  > 0 and imgs_in_album == 0
        finally:
            _db.close()
        if needs_face_repair or needs_album_repair:
            logger.warning(f"⚠️  Stale data — faces={faces_total} assigned={faces_assigned} people={people_total} imgs_in_album={imgs_in_album}/{total_imgs}")
            logger.info("🔄 Auto-repairing on startup...")
            recluster()
            logger.info("✅ Auto-repair done")
        else:
            logger.info(f"✅ DB healthy: {faces_assigned}/{faces_total} faces assigned, {people_total} people, {imgs_in_album}/{total_imgs} imgs in albums")
    except Exception as e:
        logger.warning(f"Auto-repair check failed (non-fatal): {e}")
    logger.info("✅ Ready!")
    yield


# ── app = FastAPI() MUST be here, before any app.include_router() ─────────────
app = FastAPI(title="Offline Smart Gallery API", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class FixImagePathMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        path = request.scope["path"]
        if path.startswith("/image/"):
            suffix = path[len("/image/"):]
            suffix = suffix.lstrip("/")
            if suffix.startswith("images/"):
                suffix = suffix[len("images/"):]
            bare = os.path.basename(suffix)
            if bare:
                request.scope["path"] = f"/image/{bare}"
        return await call_next(request)

app.add_middleware(FixImagePathMiddleware)

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# ── All routers registered AFTER app = FastAPI() ──────────────────────────────
app.include_router(dl_router)
app.include_router(voice_router)
app.include_router(feat_router)    # ← safely here, app already exists


# ────────────────────────────────────────────────────────────────────────────
# UTILITY
# ────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ready", "image_index": search_engine.index.ntotal if search_engine.index else 0}

@app.get("/debug")
def debug_db():
    db = SessionLocal()
    try:
        total_images         = db.query(DBImage).count()
        total_faces          = db.query(DBFace).count()
        faces_with_embedding = db.query(DBFace).filter(DBFace.face_embedding != None).count()
        faces_with_image_id  = db.query(DBFace).filter(DBFace.image_id != None).count()
        faces_with_person    = db.query(DBFace).filter(DBFace.person_id != None).count()
        total_people         = db.query(Person).count()
        total_albums         = db.query(Album).count()
        images_in_album      = db.query(DBImage).filter(DBImage.album_id != None).count()
        people_detail = []
        for p in db.query(Person).all():
            faces   = db.query(DBFace).filter(DBFace.person_id == p.id).all()
            img_ids = [f.image_id for f in faces if f.image_id]
            people_detail.append({"id": p.id, "name": p.name, "face_count": len(faces), "faces_with_image_id": len(img_ids)})
        album_detail = []
        for a in db.query(Album).all():
            imgs = db.query(DBImage).filter(DBImage.album_id == a.id).count()
            album_detail.append({"id": a.id, "title": a.title, "image_count": imgs})
        return {
            "images": total_images, "faces": total_faces,
            "faces_with_embedding": faces_with_embedding, "faces_with_image_id": faces_with_image_id,
            "faces_with_person": faces_with_person, "people": total_people,
            "albums": total_albums, "images_in_album": images_in_album,
            "people_detail": people_detail, "album_detail": album_detail,
            "action_needed": "Run POST /recluster to fix person/album assignments" if faces_with_person == 0 or images_in_album == 0 else "OK",
        }
    finally:
        db.close()

@app.get("/test-db")
def test_db():
    db = SessionLocal()
    try:
        count  = db.query(DBImage).count()
        images = db.query(DBImage).limit(1).all()
        return {
            "status": "ok", "total_images": count,
            "sample": {"filename": images[0].filename, "timestamp": images[0].timestamp.isoformat() if images and images[0].timestamp else None} if images else None,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# IMAGE LISTING & SERVING
# ────────────────────────────────────────────────────────────────────────────
@app.get("/images")
def get_all_images(limit: int = 100):
    db = SessionLocal()
    try:
        images = _live(db).limit(limit).all()
        return {"results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "face_emotion_count": img.face_emotion_count, "aesthetic_score": img.aesthetic_score, "aesthetic_rating": img.aesthetic_rating, "caption_short": img.caption_short, "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
    finally:
        db.close()

@app.get("/image/{filename:path}")
def get_image_file(filename: str):
    bare = os.path.basename(filename.lstrip("/"))
    if not bare or bare in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    candidates = [
        os.path.join(IMAGE_DIR, bare),
        os.path.normpath(os.path.join(os.getcwd(), "..", "data", "images", bare)),
        os.path.normpath(os.path.join(os.getcwd(), "data", "images", bare)),
        os.path.normpath(os.path.join(os.getcwd(), "images", bare)),
    ]
    for p in candidates:
        if os.path.exists(p):
            return FileResponse(p)
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter((DBImage.filename == bare) | (DBImage.filename == f"/images/{bare}")).first()
        if img and img.original_path:
            p = os.path.normpath(img.original_path.replace("//", "/"))
            if os.path.exists(p):
                return FileResponse(p)
            alt = os.path.join(IMAGE_DIR, os.path.basename(img.original_path))
            if os.path.exists(alt):
                return FileResponse(alt)
    finally:
        db.close()
    logger.warning(f"404 image bare={bare!r} IMAGE_DIR={IMAGE_DIR}")
    raise HTTPException(status_code=404, detail=f"Image not found: {bare}")

@app.get("/timeline")
def get_timeline():
    db = SessionLocal()
    try:
        images = _live(db).order_by(DBImage.timestamp.desc()).all()
        return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "face_emotion_count": img.face_emotion_count, "aesthetic_score": img.aesthetic_score, "aesthetic_rating": img.aesthetic_rating, "caption_short": img.caption_short, "ocr_text_enhanced": img.ocr_text_enhanced, "person_count": img.person_count, "width": img.width, "height": img.height, "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# DELETE / TRASH / RESTORE
# ────────────────────────────────────────────────────────────────────────────
@app.post("/delete_image")
def delete_image(image_id: int = Form(...)):
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        img.is_trashed = True
        img.trashed_at = datetime_module.datetime.now()
        db.commit()
        return {"success": True, "message": "Image moved to trash"}
    finally:
        db.close()

@app.post("/restore")
def restore_image(image_id: int = Form(...)):
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        img.is_trashed = False
        img.trashed_at = None
        db.commit()
        return {"success": True, "message": "Image restored"}
    finally:
        db.close()

@app.post("/permanent_delete")
def permanent_delete(image_id: int = Form(...)):
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        filename  = img.filename
        file_path = img.original_path or os.path.join(IMAGE_DIR, filename)
        bare = os.path.basename(file_path)
        for p in [file_path, os.path.join(IMAGE_DIR, bare)]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    break
                except Exception as e:
                    logger.warning(f"Could not delete file: {e}")
        if search_engine.index is not None:
            try:
                # Fast: remove just this one vector — no need to re-embed everything
                ids_arr = np.array([image_id], dtype="int64")
                search_engine.index.remove_ids(ids_arr)
                faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
                logger.info(f"✅ FAISS vector {image_id} removed")
            except Exception as e:
                logger.warning(f"FAISS remove (non-critical): {e}")
        db.query(DBFace).filter(DBFace.image_id == image_id).delete()
        db.delete(img)
        db.commit()
        return {"success": True, "message": f"Image '{filename}' permanently deleted"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/trash")
def get_trash():
    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(DBImage.is_trashed == True).all()
        return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "trashed_at": img.trashed_at.isoformat() if img.trashed_at else None, "timestamp": img.timestamp.isoformat() if img.timestamp else None, "caption": img.caption_short or ""} for img in images]}
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# FAVORITES
# ────────────────────────────────────────────────────────────────────────────
@app.post("/toggle_favorite")
def toggle_favorite(image_id: int = Form(...)):
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        img.is_favorite = not img.is_favorite
        db.commit()
        return {"success": True, "is_favorite": img.is_favorite}
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
        images = _live(db).filter(DBImage.is_favorite == True).all()
        return {"count": len(images), "results": [{"id": img.id, "filename": _img_url(img.filename), "user_tags": _user_tags(img), "photo_note": getattr(img,"photo_note","") or ""} for img in images]}
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# SEARCH
# ────────────────────────────────────────────────────────────────────────────
def _score_candidates(query_emb, sig_words, query_colors, db, top_k, extra_emb=None, text_weight=1.0, image_weight=0.0):
    if search_engine.index is None:
        return []
    if extra_emb is not None:
        blended = text_weight * query_emb + image_weight * extra_emb
        norm = np.linalg.norm(blended)
        if norm > 1e-8:
            blended /= norm
        q_vec = blended
    else:
        q_vec = query_emb
    total = search_engine.index.ntotal
    q = q_vec.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    distances, indices = search_engine.index.search(q, total)
    all_candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        clip_score = float(dist)
        if clip_score < CLIP_SCORE_MIN:
            break
        img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
        if not img or img.is_trashed:
            continue
        # (empty-caption guard removed — was killing valid results for colour queries)
        # ── OCR bonus (only if most sig_words appear) ──────────────────────
        ocr_bonus = 0.0
        ocr_src = (img.ocr_text_enhanced or "") + " " + (getattr(img, "ocr_keywords", "") or "")
        if sig_words and ocr_src.strip():
            txt  = ocr_src.lower()
            hits = sum(1 for w in sig_words if w in txt)
            # Require at least half the words to match to avoid false boosts
            if hits >= max(1, len(sig_words) * 0.5):
                ocr_bonus = min(hits / len(sig_words), 1.0)

        # ── Tag bonus ──────────────────────────────────────────────────────
        tag_bonus = 0.0
        all_tags_text = " ".join(_user_tags(img))
        if img.scene_label:
            all_tags_text += " " + img.scene_label.lower()
        if all_tags_text.strip() and any(w in all_tags_text for w in sig_words):
            tag_bonus = 1.0

        # ── Caption bonus + contradiction filter ─────────────────────────
        caption_bonus = 0.0
        hard_exclude    = False  # True = skip this result entirely
        caption_bonus   = 0.0
        caption_penalty = 0.0
        caption_src = " ".join(filter(None, [
            img.caption_short or "", img.caption_detailed or ""
        ])).lower()

        # ── Shared sets (always defined — needed even when sig_words=[]) ──
        def _stem(w):
            if len(w) > 4 and w.endswith("ses"): return w[:-2]
            if len(w) > 3 and w.endswith("ies"): return w[:-3]+"y"
            if len(w) > 3 and w.endswith("es"):  return w[:-2]
            if len(w) > 3 and w.endswith("s"):   return w[:-1]
            return w

        _COLOURS = {"black","white","red","blue","green","yellow","orange",
                    "purple","pink","brown","gray","grey","golden","silver",
                    "nude","beige","cream","navy","teal","maroon","violet"}
        _ANIMALS = {
            "horse","cow","dog","cat","bird","elephant","tiger","lion",
            "bear","wolf","deer","sheep","goat","pig","rabbit","snake",
            "fish","chicken","duck","frog","butterfly","spider",
            "fox","monkey","giraffe","zebra","panda","kangaroo","crocodile",
            "alligator","rhino","hippo","whale","dolphin","shark","eagle",
            "parrot","penguin","owl","bat","beetle","ant","bee","crab",
            "donkey","camel","llama","raccoon","squirrel","hamster",
            "mouse","rat","gecko","lizard","tortoise","turtle","swan",
            "otter","seal","walrus","beaver","meerkat","flamingo","peacock",
            "cheetah","leopard","jaguar","gorilla","chimpanzee","baboon",
            "koala","wombat","hedgehog","ferret","chinchilla","iguana",
            "chameleon","scorpion","jellyfish","octopus","lobster","shrimp",
            "kitten","puppy","calf","foal","piglet","lamb","cub","chick",
        }
        _CLOTHING = {"dress","suit","shirt","jacket","sari","skirt","gown",
                     "uniform","coat","blouse","top","hoodie","sweater",
                     "saree","kurta","lehenga","tuxedo","blazer","vest",
                     "bikini","swimsuit","robe","kimono","pyjama","shorts",
                     "trouser","pant","jeans","legging"}
        _GENDER = {"woman","girl","lady","female","man","boy","male","guy"}
        _FEMALE = {"woman","girl","lady","female"}
        _MALE   = {"man","boy","guy","male"}
        _HUMAN_WORDS = {"man","woman","boy","girl","person","people","lady",
                        "guy","child","adult","player","actor","actress"}

        # Always compute q_* from sig_words (empty set when sig_words=[])
        _q_set     = set(sig_words)
        q_colours  = _COLOURS  & _q_set
        q_animals  = _ANIMALS  & _q_set
        q_clothing = _CLOTHING & _q_set
        q_gender   = _GENDER   & _q_set

        # Always compute cap_words from caption
        if caption_src:
            _cap_raw  = caption_src.split()
            cap_words = {_stem(w) for w in _cap_raw} | set(_cap_raw)
        else:
            cap_words = set()

        if sig_words and caption_src:
            # ── Caption bonus ────────────────────────────────────────────
            hits = sum(1 for w in _q_set if w in caption_src)
            match_ratio = hits / len(_q_set)
            if match_ratio >= 0.5:
                caption_bonus = match_ratio

            # ── 1. Animal mismatch ───────────────────────────────────────
            if q_animals:
                cap_animals = _ANIMALS & cap_words
                if cap_animals and not (cap_animals & q_animals):
                    hard_exclude = True

            # ── 2. Colour + clothing contradiction (context-aware) ──────
            if not hard_exclude and q_colours:
                cap_colours  = _COLOURS & cap_words
                cap_clothing = _CLOTHING & cap_words
                same_cloth   = bool(cap_clothing & q_clothing)
                contradicting = cap_colours - q_colours
                matching      = cap_colours & q_colours
                if same_cloth and cap_colours:
                    # Caption has same clothing item → colour must match
                    # e.g. "white dress" for "black dress" → EXCLUDED
                    # "black hair wearing white dress" → "black" matches query but 
                    # the DRESS colour (white) contradicts → still exclude
                    if contradicting and not matching:
                        hard_exclude = True
                    elif contradicting and matching:
                        caption_penalty += 0.10
                elif not same_cloth and cap_colours and contradicting and not matching:
                    caption_penalty += 0.18

                        # ── 3. Clothing constraint — hard exclude on mismatch ────────
            if not hard_exclude and q_clothing:
                cap_clothing = _CLOTHING & cap_words
                if cap_clothing and not (cap_clothing & q_clothing):
                    # Caption names a DIFFERENT clothing item (e.g. "suit" for "dress")
                    hard_exclude = True
                elif not cap_clothing:
                    # Caption has NO clothing word at all
                    if q_colours and (q_colours & cap_words):
                        # Colour word appears in caption (e.g. "red eyes") but no clothing
                        # → the colour belongs to something else, not the dress → exclude
                        hard_exclude = True
                    elif q_colours and not (q_colours & cap_words):
                        # No clothing AND wrong/no colour → definitely not the right image
                        hard_exclude = True

            # ── 4. Gender mismatch ───────────────────────────────────────
            if not hard_exclude and q_gender:
                q_female = bool(_FEMALE & q_gender)
                q_male   = bool(_MALE   & q_gender)
                if q_female and (_MALE & cap_words) and not (_FEMALE & cap_words):
                    hard_exclude = True   # query=woman but caption has only men → always exclude
                if q_male and (_FEMALE & cap_words) and not (_MALE & cap_words):
                    hard_exclude = True   # query=man but caption has only women → always exclude

            if caption_penalty >= 0.40:
                hard_exclude = True
            caption_penalty = min(caption_penalty, 0.38)

        # ── Animal query: exclude human-only captions (always runs) ──────
        if not hard_exclude and q_animals and not q_gender and caption_src:
            cap_has_human  = bool(_HUMAN_WORDS & cap_words)
            cap_has_animal = bool(_ANIMALS & cap_words)
            if cap_has_human and not cap_has_animal:
                hard_exclude = True

        # ── BUGFIX: Human query: exclude animal-only captions ───────────
        if not hard_exclude and (_HUMAN_WORDS & _q_set) and caption_src:
            cap_has_human  = bool(_HUMAN_WORDS & cap_words)
            cap_has_animal = bool(_ANIMALS & cap_words)
            if cap_has_animal and not cap_has_human:
                hard_exclude = True

        # ── BUGFIX: Empty/missing caption — use scene_label as fallback ─
        # When caption_src is empty (BLIP not yet run or failed), ALL the
        # animal/human filters above are skipped because they check
        # `if sig_words and caption_src`.  Non-matching animals then leak
        # through with a pure CLIP score.  Use scene_label + user_tags to
        # catch the worst offenders.
        if not hard_exclude and not caption_src and q_animals:
            # Build a stem-aware token set from scene_label + user_tags
            _sl = (img.scene_label or "").lower()
            _ut = " ".join(_user_tags(img)).lower()
            _fb_raw = set((_sl + " " + _ut).split())
            _fb_toks = _fb_raw | {_stem(w) for w in _fb_raw}
            _fb_animals = _ANIMALS & _fb_toks
            # If scene_label names a DIFFERENT animal from the query → exclude
            if _fb_animals and not (_fb_animals & q_animals):
                hard_exclude = True
            # If scene_label has NO animal at all but has human words → exclude
            # (e.g. "person, sink" scene_label for a cat query)
            if not _fb_animals:
                _fb_humans = _HUMAN_WORDS & _fb_toks
                if _fb_humans:
                    hard_exclude = True

        # ── BUGFIX: Specific object words absent from caption ────────────────
        # If query mentions a concrete specific object (flower, rose, guitar, etc.)
        # and the caption exists but doesn't mention it at all → CLIP is hallucinating
        # semantic similarity. These concrete words almost never produce false negatives
        # (if the image has a flower, BLIP will say "flower").
        _SPECIFIC_OBJECTS = {
            "flower","rose","tulip","sunflower","daisy","bouquet","petal",
            "guitar","piano","violin","drum","trumpet","saxophone",
            "cake","pizza","burger","sandwich","coffee","wine","beer","cocktail",
            "trophy","medal","cup","award","crown",
            "balloon","umbrella","candle","lantern","flag",
            "sword","shield","arrow","bow",
            "basketball","football","soccer","tennis","cricket","baseball",
            "bicycle","motorcycle","skateboard","surfboard","snowboard",
        }
        if not hard_exclude and caption_src and sig_words:
            _q_specific = _SPECIFIC_OBJECTS & set(sig_words)
            if _q_specific:
                # Any of the specific objects must appear in caption (stemmed)
                _q_specific_stemmed = {_stem(w) for w in _q_specific} | _q_specific
                if not (_q_specific_stemmed & cap_words):
                    hard_exclude = True

        # ── Hard exclude — skip entirely ────────────────────────────────
        if hard_exclude:
            continue

        # ── Color bonus ──────────────────────────────────────────────────
        color_bonus = 0.0
        if query_colors and getattr(img, "avg_r", None) is not None:
            img_rgb = np.array([img.avg_r, img.avg_g, img.avg_b], dtype=np.float32) / 255.0
            for qc in query_colors:
                d = np.linalg.norm(img_rgb - np.array(qc, dtype=np.float32))
                color_bonus = max(color_bonus, max(0.0, 1.0 - d / np.sqrt(3)))

        # ── Final score ──────────────────────────────────────────────────
        exact_match_bonus = 0.0
        if sig_words and caption_src:
            all_present = all(w in caption_src for w in sig_words)
            majority_present = sum(1 for w in sig_words if w in caption_src) / len(sig_words) >= 0.7
            if all_present:
                exact_match_bonus = 0.25
            elif majority_present:
                exact_match_bonus = 0.12

        # Dynamic colour weight: when query explicitly asks for a colour
        # (e.g. "white horse", "red car"), pixel colour scoring should matter
        # much more than the default 0.01. Raise it to 0.15 so a brown horse
        # scores significantly lower than a white horse for "white horse".
        _colour_weight = 0.15 if query_colors else 0.01

        final_score = (0.60 * clip_score
                      + 0.20 * caption_bonus
                      + 0.15 * exact_match_bonus
                      + 0.03 * tag_bonus
                      + 0.01 * ocr_bonus
                      + _colour_weight * color_bonus
                      - caption_penalty)
        all_candidates.append({"img": img, "clip": clip_score, "final": final_score})
    all_candidates.sort(key=lambda x: x["final"], reverse=True)
    kept = [c for c in all_candidates if c["final"] >= FINAL_SCORE_MIN]
    if all_candidates:
        logger.info(f"📊 CLIP: {len(all_candidates)} above CLIP_MIN, {len(kept)} above FINAL_MIN={FINAL_SCORE_MIN}, top={all_candidates[0]['final']:.3f} bottom={all_candidates[-1]['final']:.3f}")
    results = []
    for c in kept[:top_k]:
        img = c["img"]
        results.append({
                "id": img.id,
                "filename": _img_url(img.filename),
                "score": round(c["final"] * 100, 2),
                "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
                "person_count": img.person_count or 0,
                "caption_short": img.caption_short or "",
                "caption_detailed": img.caption_detailed or "",
                "quality_score": img.quality_score or 0,
                "quality_level": img.quality_level or "",
                "sharpness": img.sharpness or 0,
                "exposure": img.exposure or 0,
                "contrast": img.contrast or 0,
                "composition": img.composition or 0,
                "dominant_emotion": img.dominant_emotion or "",
                "face_emotion_count": img.face_emotion_count or 0,
                "aesthetic_score": img.aesthetic_score or 0,
                "aesthetic_rating": img.aesthetic_rating or "",
                "ocr_text_enhanced": img.ocr_text_enhanced or "",
                "scene_label": img.scene_label or "",
                "width": img.width,
                "height": img.height,
                "is_favorite": bool(img.is_favorite),
                "user_tags": _user_tags(img),
                "photo_note": getattr(img, "photo_note", "") or "",
            })
    return results

@app.post("/search")
def search(query: str = Form(...), top_k: int = Form(20)):
    if not query or not query.strip():
        return {"status": "error", "message": "Query empty"}

    # ── Emotion search: emoji OR text keywords → filter by dominant_emotion ──
    detected_emotion = _extract_emotion_from_query(query)
    is_pure_emoji    = query.strip() in EMOTION_EMOJI_MAP

    # Check if query is MOSTLY about emotion (emotion word + optional "faces"/"photos" etc)
    EMOTION_MODIFIERS = {"faces", "face", "photos", "photo", "pictures", "picture",
                         "images", "image", "people", "person", "moments", "looking",
                         "expression", "expressions", "ones", "me", "us", "all",
                         # BUGFIX: common query prefixes that were left in non_emotion_words
                         # causing "show me happy faces" / "find sad photos" to miss the
                         # emotion route and fall through to CLIP which returns nothing.
                         "show", "find", "get", "give", "display", "share",
                         "my", "the", "a", "with", "some", "any"}
    query_words = set(query.lower().split())
    non_emotion_words = query_words - EMOTION_MODIFIERS -         {kw for kws in EMOTION_KEYWORDS.values() for kw in kws}
    is_emotion_query = detected_emotion and len(non_emotion_words) == 0

    if detected_emotion and (is_pure_emoji or is_emotion_query):
        # Route directly to emotion DB filter
        # Only match images where emotion was detected by face analysis
        # (face_emotion_count > 0 means actual faces were detected and analysed)
        db = SessionLocal()
        try:
            # Require: actual faces detected (face_emotion_count>0) AND human present (person_count>0)
            # This prevents turtles/cars/cats from appearing in happy/angry emotion search
            # Strict: only images where faces were actually detected + analysed
            # face_emotion_count > 0 = real face detection ran on this image
            imgs = _live(db).filter(
                DBImage.dominant_emotion == detected_emotion,
                DBImage.face_emotion_count > 0,
            ).order_by(DBImage.timestamp.desc()).limit(top_k).all()
            # NO fallback that removes face_emotion_count requirement.
            # Without this guard, cars/turtles/spider-man with wrongly assigned
            # emotion labels appear in results. Better to show fewer real results.
            if imgs:
                return {"status": "found", "query": query, "count": len(imgs),
                        "results": [{"id": img.id, "filename": _img_url(img.filename),
                                     "score": 90.0,
                                     "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                                     "caption_short": img.caption_short or "",
                                     "user_tags": _user_tags(img),
                                     "dominant_emotion": img.dominant_emotion,
                                     "person_count": img.person_count or 0,
                                     "photo_note": getattr(img,"photo_note","") or ""} for img in imgs]}
            else:
                # No results — emotions may not be detected yet
                return {"status": "not_found",
                        "message": f"No photos with '{detected_emotion}' emotion found. "
                                   f"Click 'Fix Emotions' in the sidebar to detect emotions on your photos."}
        finally:
            db.close()

    # ── Expand scene emojis to text for CLIP ────────────────────────────────
    expanded_query = _expand_query_emojis(query)
    # Expand sport/activity synonyms for better CLIP recall
    _words = expanded_query.lower().split()
    if any(w in SPORT_SYNONYMS for w in _words):
        _words_exp = []
        for w in _words:
            _words_exp.append(SPORT_SYNONYMS.get(w, w))
        expanded_query = " ".join(_words_exp)
    processed_query = _clean_query(expanded_query)
    logger.info(f"🔍 Search: '{query}' → '{processed_query}'")

    # BUGFIX: q_tokens_pre must be defined HERE — it is used by the
    # should_search_people pre-computation block below, which runs before
    # the later assignment at the CLIP section (line ~17609).
    q_tokens_pre = query.lower().strip().split()

    # If emotion emoji mixed with other words, boost emotion filter
    emotion_boost = detected_emotion  # may be None

    # ── Pre-compute should_search_people so sig_words can use it ────────────
    # (Full person-search logic runs later; here we just need the flag)
    _DESCRIPTOR_PRE = {
        "a","an","the","in","on","at","of","for","with","by","to","from","into",
        "is","are","was","were","be","been","has","have","had","do","does","did",
        "will","would","could","should","may","might","can","wearing","holding",
        "standing","sitting","posing","looking","long","short","tall","small",
        "big","large","little","young","old","beautiful","pretty","black","white",
        "red","blue","green","yellow","orange","purple","pink","brown","gray",
        "woman","man","girl","boy","lady","guy","person","people","female","male",
        "human","child","adult","baby","player","actor","photo","image","picture",
        "standing","next","front","back","window","field","camera","dress","suit",
        "shirt","jacket","hair","smiling",
        # FIX: query-prefix words that aren't person names
        "show","find","get","give","display","search","look","me","my","us","our",
        # FIX: emotion/state adjectives that aren't person names
        "happy","sad","angry","surprised","fearful","neutral","disgusted",
        "smiling","laughing","crying","excited","bored","tired","scared",
        # FIX: common descriptive adjectives that confused the name detector
        "funny","cute","cool","nice","great","good","bad","best","worst",
        "images","photos","pictures","moments","all","some","any","many",
    }
    # Animal/object words that look like names but are NOT people
    _NON_PERSON_WORDS = {
        # animals
        "cat","dog","horse","cow","bird","fish","fox","lion","tiger","bear","wolf",
        "pig","rabbit","duck","frog","snake","deer","sheep","goat","chicken","otter",
        "seal","panda","koala","monkey","giraffe","zebra","elephant","whale","shark",
        "eagle","owl","penguin","parrot","bee","ant","spider","butterfly","kitten",
        "puppy",
        # nature / places
        "sunset","beach","ocean","mountain","forest","river","flower","tree",
        "sky","snow","rain","night","city","park","road","bridge","building","house",
        # vehicles / objects
        "car","bus","train","plane","boat","food","pizza","cake","coffee",
        # fictional characters & universes — prevent person-search triggering
        "ironman","spiderman","batman","superman","thor","hulk","avengers",
        "marvel","anime","cartoon","fictional","hero","villain","superhero",
        "comic","spider","thanos","loki","deadpool","wolverine","aquaman",
        "flash","wonder","hawkeye","ultron","venom","groot","rocket","nebula",
        "gamora","starlord","antman","blackwidow","scarlet","vision","warmachine",
        "falcon","panther","shuri","okoye","wakanda","asgard","xmen","cyclops",
        "magneto","mystique","beast","nightcrawler","storm","phoenix","rogue",
        # music genres / scene words (not person names)
        "kpop","kdrama","jpop","bollywood","hollywood","netflix","disney","pixar",
        "hiphop","hiphop","jazz","rock","pop","metal","indie","classical",
        # common OCR / UI words that appear as image text
        "user","login","password","email","search","home","menu","settings",
        "button","click","submit","cancel","back","next","done","ok","yes","no",
        "error","loading","please","enter","select","upload","download","share",
        "follow","like","comment","post","profile","account","app","website",
        # generic concept words
        "vintage","retro","modern","classic","dark","light","minimal","abstract",
        "nature","urban","street","indoor","outdoor","close","wide","macro",
    }
    _min_len_pre      = 3 if len(q_tokens_pre) <= 3 else 4
    _nc_pre           = [w for w in q_tokens_pre if len(w) >= _min_len_pre
                         and w not in _DESCRIPTOR_PRE
                         and w not in _NON_PERSON_WORDS
                         and w.isalpha()]
    _dc_pre           = sum(1 for w in q_tokens_pre if w in _DESCRIPTOR_PRE)
    _is_desc_pre      = _dc_pre >= len(q_tokens_pre) * 0.5
    should_search_people = bool(_nc_pre) and not _is_desc_pre


    # ── Tag fast-path ─────────────────────────────────────────────────────────
    # BUGFIX: Arbitrary user tags (e.g. "birthday", "vacation", "grandma") only
    # contributed a 0.03 weight bonus to CLIP scores, making them essentially
    # invisible when typed in the search bar. Now: if ANY word in the query
    # exactly matches a known user tag in the DB, we return those tagged images
    # immediately (score=97) merged with any further CLIP/OD results.
    # This runs BEFORE the OD and CLIP paths so tags always take priority.
    _q_words_lower   = set(query.lower().strip().split())
    _tag_fast_results = []
    _tag_fast_ids     = set()
    try:
        _db_tag = SessionLocal()
        try:
            # Collect all distinct tags that exist in the DB
            _all_tag_rows = _db_tag.execute(text(
                "SELECT DISTINCT user_tags FROM images "
                "WHERE user_tags IS NOT NULL AND user_tags != '[]' "
                "AND (is_trashed IS NULL OR is_trashed=0)"
            )).fetchall()
            import json as _json_tag
            _known_tags = set()
            for _trow in _all_tag_rows:
                try:
                    for _t in _json_tag.loads(_trow[0] or "[]"):
                        _known_tags.add(_t.strip().lower())
                except Exception:
                    pass
            # Find query words that are known tags
            _matched_tags = _q_words_lower & _known_tags
            if _matched_tags:
                for _img in _live(_db_tag).all():
                    _itags = {t.strip().lower() for t in _user_tags(_img)}
                    if _itags & _matched_tags:
                        if _img.id not in _tag_fast_ids:
                            _tag_fast_ids.add(_img.id)
                            _tag_fast_results.append({
                                "id": _img.id,
                                "filename": _img_url(_img.filename) or "",
                                "score": 97.0,
                                "timestamp": _img.timestamp.isoformat() if _img.timestamp else None,
                                "caption_short": _img.caption_short or "",
                                "person_count": _img.person_count or 0,
                                "dominant_emotion": _img.dominant_emotion or "",
                                "quality_level": _img.quality_level or "",
                                "quality_score": _img.quality_score or 0,
                                "aesthetic_score": _img.aesthetic_score or 0,
                                "user_tags": _user_tags(_img),
                                "photo_note": getattr(_img, "photo_note", "") or "",
                                "is_favorite": bool(_img.is_favorite),
                                "scene_label": _img.scene_label or "",
                                "width": _img.width, "height": _img.height,
                            })
                logger.info(f"🏷️ Tag fast-path: matched tags {_matched_tags} → {len(_tag_fast_results)} images")
        finally:
            _db_tag.close()
    except Exception as _tag_err:
        logger.warning(f"Tag fast-path failed (non-fatal): {_tag_err}")

    # ── Object-detection label fast-path ──────────────────────────────────
    # Check if any query word directly maps to a COCO-detected object.
    # scene_label contains comma-separated Faster R-CNN detections (e.g. "cat, person").
    # This is far more reliable than CLIP for specific animals/objects.
    QUERY_TO_COCO = {
        "cat":"cat","cats":"cat","kitten":"cat","kitty":"cat","kittens":"cat",
        "dog":"dog","dogs":"dog","puppy":"dog","puppies":"dog","pup":"dog",
        "horse":"horse","horses":"horse","pony":"horse",
        "bird":"bird","birds":"bird","parrot":"bird","eagle":"bird","owl":"bird",
        "cow":"cow","cows":"cow","bull":"cow","cattle":"cow",
        "elephant":"elephant","bear":"bear","zebra":"zebra","giraffe":"giraffe",
        "sheep":"sheep","goat":"sheep","deer":"deer","rabbit":"rabbit",
        "soccer":"sports ball","football":"sports ball","sport":"sports ball",
        "cricket":"baseball bat","baseball":"baseball bat",
        "tennis":"tennis racket","racket":"tennis racket",
        "surfing":"surfboard","surf":"surfboard","skateboard":"skateboard",
        "skiing":"skis","ski":"skis","snowboard":"snowboard",
        "car":"car","cars":"car","truck":"truck","bus":"bus",
        "motorcycle":"motorcycle","bicycle":"bicycle","bike":"bicycle",
        "boat":"boat","ship":"boat","train":"train",
        "airplane":"airplane","plane":"airplane","aircraft":"airplane",
        "pizza":"pizza","cake":"cake","sandwich":"sandwich",
        "apple":"apple","banana":"banana","orange":"orange",
        "bottle":"bottle","cup":"cup","wine":"wine glass",
        "laptop":"laptop","keyboard":"keyboard","mouse":"mouse",
        "phone":"cell phone","cellphone":"cell phone",
        "chair":"chair","sofa":"couch","couch":"couch","bed":"bed",
        "tv":"tv","television":"tv","monitor":"tv",
        "book":"book","books":"book","clock":"clock","vase":"vase",
        "scissors":"scissors","knife":"knife","fork":"fork","spoon":"spoon",
    }
    _q_lower_words = set(query.lower().split())
    _coco_targets = set()
    for _w in _q_lower_words:
        if _w in QUERY_TO_COCO:
            _coco_targets.add(QUERY_TO_COCO[_w])
    
    if _coco_targets and not should_search_people:
        # Direct object detection search — bypass CLIP entirely for these queries
        _db_od = SessionLocal()
        try:
            _od_results = []
            _seen_od    = set()   # dedup by image id
            _seen_fn    = set()   # dedup by filename (catches re-indexed duplicates)

            # Broad animal vocabulary used by BLIP captions — used below to
            # verify that the COCO model's scene_label isn't a misclassification.
            _BROAD_ANIMALS = {
                "horse","cow","dog","cat","bird","elephant","tiger","lion","bear",
                "wolf","deer","sheep","goat","pig","rabbit","snake","fish","chicken",
                "duck","frog","fox","monkey","giraffe","zebra","panda","otter",
                "squirrel","raccoon","hamster","mouse","rat","kitten","puppy","calf",
                "foal","piglet","lamb","cub","chick","parrot","penguin","owl","bat",
                "seal","beaver","meerkat","flamingo","peacock","cheetah","leopard",
                "jaguar","gorilla","koala","hedgehog","iguana","chameleon","scorpion",
                "jellyfish","octopus","lobster","crab","shrimp","bison","buffalo",
                "moose","elk","reindeer","caribou","hyena","wildebeest","gazelle",
            }

            def _od_row(img, score):
                return {
                    "id": img.id,
                    "filename": _img_url(img.filename),
                    "score": score,
                    "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                    "caption_short": img.caption_short or "",
                    "person_count": img.person_count or 0,
                    "dominant_emotion": img.dominant_emotion or "",
                    "quality_level": img.quality_level or "",
                    "quality_score": img.quality_score or 0,
                    "aesthetic_score": img.aesthetic_score or 0,
                    "user_tags": _user_tags(img),
                    "photo_note": getattr(img, "photo_note", "") or "",
                    "is_favorite": bool(img.is_favorite),
                }

            for _img in _live(_db_od).all():
                fn = _img_url(_img.filename) or ""
                # ── BUGFIX: dedup by filename to catch re-indexed duplicates ──
                if _img.id in _seen_od or fn in _seen_fn:
                    continue

                # ── BUGFIX: also match via user_tags (manual tags bypass OD) ──
                _img_tags = {t.strip().lower() for t in _user_tags(_img)}
                is_tag_match = bool(_img_tags & _coco_targets)

                # ── scene_label match (original OD path) ─────────────────────
                is_od_match = False
                _match_score = 0
                if _img.scene_label:
                    _detected = {x.strip().lower() for x in _img.scene_label.split(",")}
                    _match_score = sum(1 for t in _coco_targets if t in _detected)
                    is_od_match = _match_score > 0

                if not is_od_match and not is_tag_match:
                    continue

                # ── BUGFIX: caption-verify OD matches to filter COCO misclassifications ──
                # Faster R-CNN doesn't know fox/otter/squirrel/rabbit/panda (not COCO
                # classes). It often mislabels them as "cat" or "dog". BLIP captions
                # are far more accurate — if the caption names a DIFFERENT animal,
                # the scene_label detection is a false positive. Skip it.
                # Exception: user-tagged images are always trusted.
                if is_od_match and not is_tag_match:
                    _cap = (_img.caption_short or "").lower()
                    if _cap:
                        _cap_toks_raw = set(_cap.split())
                        def _stem_tok(w):
                            if len(w) > 4 and w.endswith("ses"): return w[:-2]
                            if len(w) > 4 and w.endswith("ies"): return w[:-3]+"y"
                            if len(w) > 3 and w.endswith("es"):  return w[:-2]
                            if len(w) > 3 and w.endswith("s"):   return w[:-1]
                            return w
                        _cap_toks = _cap_toks_raw | {_stem_tok(w) for w in _cap_toks_raw}
                        _cap_animals = _BROAD_ANIMALS & _cap_toks
                        if _cap_animals and not (_cap_animals & _coco_targets):
                            logger.info(
                                f"🚫 OD-filter: id={_img.id} caption has {_cap_animals}, "
                                f"not {_coco_targets} — skipping misclassification"
                            )
                            continue
                    else:
                        # Caption empty/missing — fall back to scene_label cross-check.
                        # If scene_label contains ONLY human labels and no animal,
                        # the "cat"/"dog" detection is almost certainly a false positive.
                        _sl_raw = (_img.scene_label or "").lower()
                        _sl_toks = set(_sl_raw.replace(",", " ").split())
                        _sl_animals = _BROAD_ANIMALS & _sl_toks
                        _HUMAN_SL   = {"person", "man", "woman", "boy", "girl", "people"}
                        _sl_humans  = _HUMAN_SL & _sl_toks
                        if _sl_humans and not _sl_animals:
                            logger.info(
                                f"🚫 OD-filter (no caption): id={_img.id} scene_label has "
                                f"only humans {_sl_humans}, not {_coco_targets} — skipping"
                            )
                            continue

                # FIX Bug2: caption exists, no animals detected, but query IS for an animal
                # e.g. "two men in suits" for a "horse" query → exclude
                if is_od_match and not is_tag_match and _cap:
                    _cap_toks_check = {w.rstrip("s") if len(w) > 3 else w
                                       for w in set(_cap.split())} | set(_cap.split())
                    _cap_animals_check = _BROAD_ANIMALS & _cap_toks_check
                    _animal_targets = _coco_targets & {
                        "cat","dog","horse","bird","cow","elephant","bear",
                        "zebra","giraffe","sheep","deer","rabbit","horse",
                    }
                    if _animal_targets and not _cap_animals_check:
                        # Caption has NO animal — check if it's a human-only image
                        _HUMAN_CAP = {"man","woman","person","girl","boy","people","men",
                                      "women","actor","actress","player","character","he",
                                      "she","they","guy","lady","child","adult"}
                        if _HUMAN_CAP & _cap_toks_check:
                            logger.info(
                                f"🚫 OD-filter (human-only caption): id={_img.id} "
                                f"caption '{_cap[:60]}' has no animal for {_animal_targets}"
                            )
                            continue

                # FIX Bug4: non-animal OD targets (car, truck, etc.) with unrelated captions
                # e.g. "thor and thor in the avengers movie" for a "red car" query → exclude
                if is_od_match and not is_tag_match and _cap:
                    _VEHICLE_COCO = {"car","truck","bus","motorcycle","bicycle","boat",
                                     "airplane","train"}
                    _vehicle_targets = _coco_targets & _VEHICLE_COCO
                    if _vehicle_targets:
                        _VEHICLE_WORDS = {"car","truck","bus","motorcycle","bike","bicycle",
                                          "vehicle","driving","road","highway","parking",
                                          "train","boat","ship","airplane","plane","suv",
                                          "sedan","sports","jeep","porsche","ferrari",
                                          "bmw","mercedes","driving","auto","automobile"}
                        _FICTIONAL_WORDS = {"marvel","avengers","thor","ironman","iron",
                                            "spiderman","batman","superman","hulk","anime",
                                            "cartoon","movie","film","fictional","hero",
                                            "villain","superhero","comic","bust","armor",
                                            "avenger","character","suit"}
                        _cap_set_v = set(_cap.split())
                        _has_vehicle_word = bool(_VEHICLE_WORDS & _cap_set_v)
                        _has_fictional    = bool(_FICTIONAL_WORDS & _cap_set_v)
                        if not _has_vehicle_word and _has_fictional:
                            logger.info(
                                f"🚫 OD-filter (fictional caption for vehicle): id={_img.id} "
                                f"caption '{_cap[:60]}'"
                            )
                            continue

                score = 92.0 if is_tag_match else round(85.0 + _match_score * 5, 1)

                # ── BUGFIX: Colour-word filter in OD path ─────────────────────
                # The OD fast-path matches by object class (horse/cat/car) but
                # completely ignores colour words in the query.
                # "white horse" was returning brown/dark horses because OD just
                # found all horses and returned them all at score 85–90.
                #
                # Rules (only when query contains colour words):
                #   1. Caption has colour that CONTRADICTS query → EXCLUDE
                #      e.g. "brown horse" for "white horse" → skip
                #   2. Caption has colour that MATCHES query → KEEP (boosted)
                #   3. Caption exists but has NO colour word:
                #      → Don't return from OD; fall through to CLIP which uses
                #        pixel-level colour scoring (avg_r/g/b) — much more reliable
                #        for unlabelled subjects
                #   4. No caption yet (BLIP not run) → KEEP (can't verify)
                if not is_tag_match and _q_lower_words:
                    _COLOUR_WORDS = {
                        "white","black","red","blue","green","yellow","orange",
                        "purple","pink","brown","gray","grey","golden","silver",
                        "dark","light","pale","cream","beige","spotted","striped",
                    }
                    _q_colours = _COLOUR_WORDS & _q_lower_words
                    if _q_colours:
                        _cap_for_colour = (_img.caption_short or "").lower()
                        if _cap_for_colour:  # only apply filter if caption exists
                            _cap_toks_c = set(_cap_for_colour.split())
                            _cap_colours = _COLOUR_WORDS & _cap_toks_c
                            _matching     = _q_colours & _cap_colours
                            _contradicting = _cap_colours - _q_colours

                            # "black and white photo/photograph" = monochrome style,
                            # not a colour description of the subject → don't exclude
                            _is_mono = (("black" in _cap_toks_c or "white" in _cap_toks_c) and
                                        ("photo" in _cap_toks_c or "photograph" in _cap_toks_c
                                         or "drawing" in _cap_toks_c or "painting" in _cap_toks_c))

                            if _cap_colours and not _matching and not _is_mono:
                                # Caption names a DIFFERENT colour → wrong subject
                                logger.info(
                                    f"🎨 OD colour-filter: id={_img.id} "
                                    f"query_colours={_q_colours} cap_colours={_cap_colours} → skip"
                                )
                                continue
                            elif not _cap_colours:
                                # Caption has no colour — defer to CLIP+pixel scoring
                                # Don't add to OD results; let it fall through to CLIP
                                logger.debug(
                                    f"🎨 OD colour-defer: id={_img.id} "
                                    f"no colour in caption → CLIP will score by pixel"
                                )
                                continue
                            elif _matching:
                                # Confirmed colour match → boost score
                                score = min(score + 5, 97.0)

                _seen_od.add(_img.id)
                _seen_fn.add(fn)
                _od_results.append(_od_row(_img, score))

            # Sort by score desc
            _od_results.sort(key=lambda x: x["score"], reverse=True)

            # FIX Bug1: dedup by caption text — catches same-content images uploaded
            # twice (different UUID filenames, so filename-dedup misses them).
            _seen_cap_text = set()
            _od_deduped    = []
            for _r in _od_results:
                _ct = (_r.get("caption_short") or "").strip().lower()
                if _ct and _ct in _seen_cap_text:
                    logger.info(f"🔁 OD dedup: skipping duplicate caption '{_ct[:50]}'")
                    continue
                if _ct:
                    _seen_cap_text.add(_ct)
                _od_deduped.append(_r)
            _od_results = _od_deduped

            if _od_results:
                logger.info(f"✅ OD-search '{query}' → {len(_od_results)} results via scene_label/tags")
                return {
                    "status": "found",
                    "query": query,
                    "count": len(_od_results),
                    "results": _od_results[:top_k],
                }
        finally:
            _db_od.close()
        # OD found nothing → try caption text search for the animal/object words
        _caption_results = []
        _seen_cap = set()
        # Search caption text for query words directly
        _search_terms = list(_q_lower_words) + [v for k,v in QUERY_TO_COCO.items() if k in _q_lower_words]
        _search_terms = list(set(_search_terms))
        _db_cap = SessionLocal()
        try:
            for _img in _live(_db_cap).all():
                # Search caption, scene_label, AND user tags
                _cap_lower = (_img.caption_short or "").lower()
                _det_lower = (_img.caption_detailed or "").lower()
                _scene_lower = (_img.scene_label or "").lower()
                _tags_lower = " ".join(_user_tags(_img)).lower()
                _haystack = " ".join(filter(None,[_cap_lower, _det_lower, _scene_lower, _tags_lower]))
                # Check if any search term appears anywhere
                if _haystack and any(t in _haystack for t in _search_terms) and _img.id not in _seen_cap:
                    _seen_cap.add(_img.id)
                    _caption_results.append({
                        "id": _img.id,
                        "filename": _img_url(_img.filename),
                        "score": 80.0,
                        "timestamp": _img.timestamp.isoformat() if _img.timestamp else None,
                        "caption_short": _img.caption_short or "",
                        "person_count": _img.person_count or 0,
                        "dominant_emotion": _img.dominant_emotion or "",
                        "quality_level": _img.quality_level or "",
                        "quality_score": _img.quality_score or 0,
                        "aesthetic_score": _img.aesthetic_score or 0,
                        "user_tags": _user_tags(_img),
                        "photo_note": getattr(_img, "photo_note", "") or "",
                        "is_favorite": bool(_img.is_favorite),
                    })
        finally:
            _db_cap.close()
        if _caption_results:
            logger.info(f"✅ Caption-search '{query}' → {len(_caption_results)} results")
            return {
                "status": "found", "query": query,
                "count": len(_caption_results),
                "results": sorted(_caption_results, key=lambda x: x["score"], reverse=True)[:top_k]
            }
        logger.info(f"⚠️ OD+Caption search found nothing for {_search_terms}, falling back to CLIP")

    query_emb = search_engine.get_text_embedding(processed_query, use_prompt_ensemble=True)
    if query_emb is None or search_engine.index is None:
        return {"status": "error", "message": "No images indexed. Upload some photos first."}
    query_lower  = query.lower()
    # q_tokens_pre already defined above (before should_search_people pre-compute)

    # For person-name queries, clear sig_words so CLIP scoring
    # doesn't give caption bonus to irrelevant images (e.g. pig-in-sink for "Sadie Sink")
    sig_words    = [] if should_search_people else [w for w in processed_query.lower().split() if len(w) > 2]
    query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in query_lower]

    # ── Compound word expansion ────────────────────────────────────────────────
    # BLIP captions write "iron man" (two words) but users search "ironman" (one).
    # Expand known compounds so all keyword paths find the image.
    _COMPOUND_SPLITS = {
        "ironman":     ["iron","man"],      "spiderman":    ["spider","man"],
        "batman":      ["bat","man"],       "superman":     ["super","man"],
        "blackwidow":  ["black","widow"],   "antman":       ["ant","man"],
        "warmachine":  ["war","machine"],   "blackpanther": ["black","panther"],
        "ironheart":   ["iron","heart"],    "deadpool":     ["dead","pool"],
        "daredevil":   ["dare","devil"],    "spiderverse":  ["spider","verse"],
    }
    if sig_words:
        _exp = []
        for _sw in sig_words:
            _exp.append(_sw)
            if _sw in _COMPOUND_SPLITS:
                _exp.extend(_COMPOUND_SPLITS[_sw])
        sig_words = list(dict.fromkeys(_exp))

    # Also expand the processed_query itself so CLIP gets "iron man" instead of "ironman"
    _pq_words = processed_query.lower().split()
    _pq_expanded = []
    for _pw in _pq_words:
        if _pw in _COMPOUND_SPLITS:
            _pq_expanded.extend(_COMPOUND_SPLITS[_pw])
        else:
            _pq_expanded.append(_pw)
    if _pq_expanded != _pq_words:
        processed_query = " ".join(_pq_expanded)
        logger.info(f"🔤 Compound expand: '{query}' → CLIP query='{processed_query}'")
    db = SessionLocal()
    try:
        results = _score_candidates(query_emb, sig_words, query_colors, db, top_k)
        # ── Person / actor name search ─────────────────────────────────────
        # ── Person name search — only for name-like queries ────────────────
        # CRITICAL: Only run if query looks like a person name, NOT for
        # descriptive queries like "woman in black dress".
        # Rules:
        #   - Must have at least one word >= 4 chars that isn't a common word
        #   - Must not be a purely descriptive query (too many common words)
        #   - Person name matching uses WHOLE WORD matching only (not substring)

        DESCRIPTOR_WORDS = {
            # articles / prepositions / conjunctions
            "a","an","the","in","on","at","of","for","with","by","to","from",
            "into","onto","over","under","above","below","near","next","beside",
            "and","or","but","not","nor","yet","so","both","either","neither",
            # verbs
            "is","are","was","were","be","been","being","has","have","had",
            "do","does","did","will","would","could","should","may","might",
            "can","wearing","holding","standing","sitting","posing","looking",
            "running","walking","smiling","laughing","dancing","jumping","riding",
            # common adjectives
            "long","short","tall","small","big","large","little","young","old",
            "beautiful","pretty","lovely","nice","great","black","white","red",
            "blue","green","yellow","orange","purple","pink","brown","gray",
            "dark","light","bright","dark","shiny","fluffy","cute","funny",
            # people descriptors (not names)
            "woman","man","girl","boy","lady","guy","person","people","female",
            "male","human","child","adult","baby","teenager","player","actor",
            # scene/object words — CRITICAL: prevents these being treated as names
            "photo","image","picture","standing","next","another","front","back",
            "window","field","camera","dress","suit","shirt","jacket","hair",
            "wearing","sitting","holding","looking","posing","smiling",
            # common household/body nouns that appear in captions
            "sink","table","chair","floor","wall","door","room","house","home",
            "hand","face","head","hair","eyes","mouth","nose","body","back",
            "water","grass","tree","flower","rock","snow","sand","road","street",
            "book","phone","glass","bowl","plate","ball","box","bag","hat",
            "food","cake","pizza","coffee","wine","beer","milk","soup",
            "park","beach","mountain","forest","city","stage","roof","pool",
            "movie","film","show","song","music","sport","game","team","show",
            "scene","time","year","day","night","morning","evening","world",
        }

        raw_q      = query.lower().strip()
        q_tokens   = raw_q.split()
        # Words that could plausibly be part of a name (long, not descriptive)
        # For short queries (≤3 words like "tom holland"), allow 3-char words
        min_name_len = 3 if len(q_tokens) <= 3 else 4
        name_candidates = [
            w for w in q_tokens
            if len(w) >= min_name_len
            and w not in DESCRIPTOR_WORDS
            and w.isalpha()
        ]

        descriptor_count = sum(1 for w in q_tokens if w in DESCRIPTOR_WORDS)
        is_descriptive   = descriptor_count > len(q_tokens) * 0.5   # strict: >50% descriptors needed
        should_search_people = bool(name_candidates) and not is_descriptive

        vector_ids   = {r["id"] for r in results}
        people_rows  = db.query(Person).all() if should_search_people else []

        # ── 1. Match against named person clusters ────────────────────────
        matched_people = []
        if should_search_people:
            for p in people_rows:
                pname_raw = (p.name or "").strip()
                pn        = pname_raw.lower()
                if not pn:
                    continue
                # Skip default "Person N" labels
                if re.match(r"^person\s+\d+$", pn):
                    continue
                # Skip OCR garbage: all-caps multi-word, or contains digits,
                # or contains non-alpha chars (PETRONAS INEQS, Hak IINDIA, etc.)
                if re.search(r"\d", pname_raw):
                    continue   # contains digit → OCR garbage
                p_words = pn.split()
                # If all words are UPPERCASE in original → likely OCR label not a name
                if all(w == w.upper() and len(w) > 1 for w in pname_raw.split()):
                    continue
                # Whole-word matching ONLY — "in" must NOT match inside "india"
                # Use word boundary: check if name word appears as standalone token
                p_parts = [pt for pt in p_words if len(pt) >= 3]
                query_tokens_set = set(q_tokens)
                matched = False
                for pt in p_parts:
                    if pt in query_tokens_set:        # exact token match
                        matched = True; break
                for nc in name_candidates:
                    if nc in p_words:                 # query name-word in person name
                        matched = True; break
                    if pn == nc or nc == pn:
                        matched = True; break
                    # Allow partial first/last name: "vijay" matches "vijay kumar"
                    if any(nc == pw for pw in p_words):
                        matched = True; break
                if matched:
                    matched_people.append(p)

            # ── 2. Caption / OCR / VQA text search (works WITHOUT renaming) ──
            # Only run if face-cluster lookup found no results yet
            # (avoids flooding with false positives from common words in OCR)
            COMMON_WORDS = {
                "the","and","for","are","but","not","you","all","any","can",
                "had","her","was","one","our","out","day","get","has","him",
                "his","how","its","may","new","now","old","see","two","way",
                "who","did","man","men","two","use","she","him","this","that",
                "with","from","they","have","more","will","home","also","into",
                "over","time","very","when","your","come","here","just","like",
                "long","make","many","most","some","them","than","then","these",
                "well","were","what","each","much","both","been","only","same",
                "india","team","cup","match","player","game","year","last",
            }
            # Only do caption search for proper-noun-looking queries
            # (single words >= 4 chars that aren't common English words)
            search_words = [w for w in name_candidates if len(w) >= 4 and w not in COMMON_WORDS]
            if search_words and not matched_people:
                for img in _live(db).all():
                    if img.id in vector_ids:
                        continue
                    vqa_person = ""
                    try:
                        vqa_d = json.loads(img.caption_vqa or "{}")
                        vqa_person = vqa_d.get("person", "") or vqa_d.get("subject", "")
                    except Exception:
                        pass
                    hay = " ".join(filter(None, [
                        img.caption_short or "",
                        img.caption_detailed or "",
                        img.ocr_text_enhanced or "",
                        img.scene_label or "",
                        vqa_person,
                        " ".join(_user_tags(img)),
                    ])).lower()
                    # Require ALL search words to match (avoids false positives)
                    if all(w in hay for w in search_words):
                        results.append({
                            "id": img.id, "filename": _img_url(img.filename),
                            "score": 65.0,
                            "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                            "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
                            "person_count": img.person_count or 0,
                            "caption_short": img.caption_short or "",
                            "user_tags": _user_tags(img),
                            "photo_note": getattr(img, "photo_note", "") or "",
                        })
                        vector_ids.add(img.id)

            if matched_people:
                # CRITICAL: When we found a person match, REPLACE CLIP results entirely.
                # CLIP results for name queries are unreliable (e.g. "sadie sink" → pig-in-sink).
                # Only return face-cluster matches + caption/OCR name matches.
                person_results = []
                person_ids_seen = set()

                for person in matched_people:
                    logger.info(f"🧑 Searching for person: '{person.name}' (id={person.id})")
                    face_records = db.query(DBFace).filter(DBFace.person_id == person.id).all()
                    face_img_ids = list({f.image_id for f in face_records if f.image_id})
                    logger.info(f"  → {len(face_records)} face records, {len(face_img_ids)} unique image_ids")

                    # 1. Face-cluster matched images (highest confidence = 95)
                    if face_img_ids:
                        for img in _live(db).filter(DBImage.id.in_(face_img_ids)).all():
                            if img.id not in person_ids_seen:
                                person_results.append({
                                    "id": img.id, "filename": _img_url(img.filename),
                                    "score": 95.0,
                                    "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                                    "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
                                    "person_count": img.person_count or 0,
                                    "caption_short": img.caption_short or "",
                                    "user_tags": _user_tags(img),
                                    "photo_note": getattr(img, "photo_note", "") or "",
                                    "matched_person": person.name,
                                })
                                person_ids_seen.add(img.id)

                    # 2. Caption/OCR text search for the FULL NAME as a phrase only
                    # Use full name phrase — prevents "sink" matching pig captions
                    name_kw = person.name.lower()
                    name_parts = name_kw.split()  # ["sadie", "sink"]
                    for img in _live(db).all():
                        if img.id in person_ids_seen:
                            continue
                        hay = " ".join(filter(None, [
                            img.caption_short or "", img.caption_detailed or "",
                            img.caption_vqa or "", img.ocr_text_enhanced or "",
                            img.scene_label or ""
                        ])).lower()
                        # Must contain FULL NAME as phrase OR all parts together
                        # NOT individual words (prevents "sink" matching pig captions)
                        if name_kw in hay or (len(name_parts) > 1 and all(p in hay for p in name_parts)):
                            # Extra check: if name has common words (like "sink"),
                            # require the rare part to appear near human context
                            person_results.append({
                                "id": img.id, "filename": _img_url(img.filename),
                                "score": 75.0,
                                "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                                "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None,
                                "person_count": img.person_count or 0,
                                "caption_short": img.caption_short or "",
                                "user_tags": _user_tags(img),
                                "photo_note": getattr(img, "photo_note", "") or "",
                                "matched_person": person.name,
                            })
                            person_ids_seen.add(img.id)

                    if not face_img_ids:
                        logger.warning(f"  ⚠️ No face records for '{person.name}' — only caption search. Run Re-index AI.")

                if person_results:
                    results = sorted(person_results, key=lambda x: x["score"], reverse=True)
                    return {"status": "found", "query": query, "count": len(results), "results": results[:top_k]}
                else:
                    # Person found in DB but no photos linked yet — tell user to re-index
                    names = ", ".join(p.name for p in matched_people)
                    return {
                        "status": "not_found",
                        "message": f"Found person '{names}' but no photos linked. Click Re-index AI to rebuild face clusters.",
                        "people_suggestions": [{"id": p.id, "name": p.name, "cover": None} for p in matched_people],
                    }

        # ── No person found in DB for a name query → try content search first ──
        if should_search_people and not matched_people:
            # Before giving up, attempt 3 fallbacks in priority order:
            # 1. Tag fast-path results (user manually tagged this image)
            # 2. Caption/OCR keyword search (works for ironman, thor, spider, etc.)
            # 3. Only if both fail → show "not found" with people suggestions

            # ── Fallback 1: tags ──────────────────────────────────────────────
            if _tag_fast_results:
                logger.info(f"🏷️ Person-path fallback → {len(_tag_fast_results)} tag results for '{query}'")
                return {
                    "status": "found", "query": query,
                    "count": len(_tag_fast_results),
                    "results": _tag_fast_results[:top_k],
                }

            # ── Fallback 2: caption + OCR keyword search ──────────────────────
            # Covers: "ironman", "thor", "spider", "kpop", "user" (OCR), etc.
            # Use the original query words (NOT processed_query which strips non-CLIP terms)
            _kw_query_words = [w for w in query.lower().split() if len(w) > 2]
            _kw_results_fb  = []
            _kw_seen_fb     = set()
            _kw_seen_cap_fb = set()
            if _kw_query_words:
                for _img_fb in _live(db).all():
                    if _img_fb.id in _kw_seen_fb:
                        continue
                    # Build search haystack: caption + OCR + scene_label + tags
                    _fb_hay = " ".join(filter(None, [
                        (_img_fb.caption_short    or "").lower(),
                        (_img_fb.caption_detailed or "").lower(),
                        (_img_fb.ocr_text_enhanced or "").lower(),
                        (_img_fb.scene_label      or "").lower(),
                        " ".join(_user_tags(_img_fb)).lower(),
                    ]))
                    if not _fb_hay:
                        continue
                    # ALL query words must appear (strict — avoids false positives)
                    if all(w in _fb_hay for w in _kw_query_words):
                        _ct_fb = (_img_fb.caption_short or "").strip().lower()
                        if _ct_fb and _ct_fb in _kw_seen_cap_fb:
                            continue
                        if _ct_fb:
                            _kw_seen_cap_fb.add(_ct_fb)
                        _kw_seen_fb.add(_img_fb.id)
                        _kw_results_fb.append({
                            "id": _img_fb.id,
                            "filename": _img_url(_img_fb.filename) or "",
                            "score": 80.0,
                            "timestamp": _img_fb.timestamp.isoformat() if _img_fb.timestamp else None,
                            "caption_short": _img_fb.caption_short or "",
                            "person_count": _img_fb.person_count or 0,
                            "dominant_emotion": _img_fb.dominant_emotion or "",
                            "quality_level": _img_fb.quality_level or "",
                            "quality_score": _img_fb.quality_score or 0,
                            "aesthetic_score": _img_fb.aesthetic_score or 0,
                            "user_tags": _user_tags(_img_fb),
                            "photo_note": getattr(_img_fb, "photo_note", "") or "",
                            "is_favorite": bool(_img_fb.is_favorite),
                            "scene_label": _img_fb.scene_label or "",
                            "width": _img_fb.width, "height": _img_fb.height,
                        })
            if _kw_results_fb:
                logger.info(f"📝 Person-path caption/OCR fallback → {len(_kw_results_fb)} results for '{query}'")
                return {
                    "status": "found", "query": query,
                    "count": len(_kw_results_fb),
                    "results": sorted(_kw_results_fb, key=lambda x: x["score"], reverse=True)[:top_k],
                }

            # ── Fallback 3: truly nothing found → people suggestions ──────────
            all_people = db.query(Person).all()
            suggestions = []
            for p in all_people:
                if re.match(r"^person\s+\d+$", (p.name or "").lower()): continue
                face_recs = db.query(DBFace).filter(DBFace.person_id == p.id).limit(1).all()
                if face_recs and face_recs[0].image_id:
                    cover_img = db.query(DBImage).filter(DBImage.id == face_recs[0].image_id).first()
                    if cover_img:
                        suggestions.append({"id": p.id, "name": p.name, "cover": _img_url(cover_img.filename)})
            return {
                "status": "not_found",
                "message": f"No photos matched '{query}'.",
                "people_suggestions": suggestions[:12],
            }

        # Fallback: discard any CLIP junk when person matched
        if matched_people:
            person_results = [r for r in results if r.get("matched_person")]
            if person_results:
                results = person_results
            results.sort(key=lambda x: x["score"], reverse=True)

        # ── Skip keyword fallback when we already have person results ──────────
        # Person name searches should ONLY return person face images, not
        # generic CLIP results or keyword matches for ambiguous words like "sink"
        skip_kw_fallback = bool(matched_people)

        if sig_words and not skip_kw_fallback:
            vector_ids = {r["id"] for r in results}
            KW_STOP = {"the","a","an","in","on","at","of","for","with","by","to",
                       "from","is","are","was","were","be","and","or","but","not"}
            meaningful = [w for w in sig_words if w not in KW_STOP and len(w) > 2]

            # ── Rebuild contradiction context for filtering ────────────────
            COLOURS_KW = {"black","white","red","blue","green","yellow","orange",
                          "purple","pink","brown","gray","grey","golden","silver"}
            CLOTHING_KW = {"dress","suit","shirt","jacket","sari","skirt","gown",
                           "uniform","coat","blouse","saree","lehenga"}
            ANIMALS_KW = {"horse","cow","dog","cat","bird","fox","tiger","lion",
                          "bear","wolf","pig","rabbit","fish","duck","frog","otter",
                          "elephant","monkey","deer","sheep","goat","snake","chicken",
                          "kitten","puppy","calf","hamster","squirrel","raccoon"}
            FEMALE_KW = {"woman","girl","lady","female"}
            MALE_KW   = {"man","boy","guy","male"}
            q_set_kw      = set(meaningful)
            q_colours_kw  = COLOURS_KW  & q_set_kw
            q_clothing_kw = CLOTHING_KW & q_set_kw
            q_animals_kw  = ANIMALS_KW  & q_set_kw
            q_gender_kw   = (FEMALE_KW | MALE_KW) & q_set_kw

            def _kw_ok(caption_text):
                """Return True if caption doesn't contradict the query."""
                cap = set(caption_text.split())
                # Stemmed cap words
                def s(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
                cap_stem = {s(w) for w in cap} | cap

                if q_animals_kw:
                    cap_animals = ANIMALS_KW & cap_stem
                    if cap_animals and not (cap_animals & q_animals_kw):
                        return False   # different animal
                    # Also: if caption has PEOPLE words but NO animals at all → exclude
                    # (e.g. "man standing in a field" shouldn't appear for "dog" query)
                    PEOPLE_KW = {"man","woman","person","girl","boy","people","guy",
                                 "lady","male","female","player","actor","celebrity"}
                    if not cap_animals and (PEOPLE_KW & cap_stem):
                        return False   # people-only caption for animal query
                if q_colours_kw:
                    cap_col = COLOURS_KW & cap
                    contradicting = cap_col - q_colours_kw
                    matching      = cap_col & q_colours_kw
                    if contradicting and not matching:
                        return False   # only wrong colours
                if q_clothing_kw:
                    cap_cl = CLOTHING_KW & cap_stem
                    if cap_cl and not (cap_cl & q_clothing_kw):
                        return False   # different clothing
                if q_gender_kw:
                    q_f = bool(FEMALE_KW & q_gender_kw)
                    q_m = bool(MALE_KW   & q_gender_kw)
                    if q_f and (MALE_KW & cap) and not (FEMALE_KW & cap):
                        return False
                    if q_m and (FEMALE_KW & cap) and not (MALE_KW & cap):
                        return False
                return True

            kw_candidates = _live(db).filter(
                DBImage.caption_short.isnot(None) | DBImage.caption_detailed.isnot(None)
            ).all()
            for img in kw_candidates:
                if img.id in vector_ids:
                    continue
                caption_text = " ".join(filter(None,[
                    img.caption_short or "", img.caption_detailed or ""
                ])).lower()
                haystack = caption_text + " " + " ".join(filter(None,[
                    img.ocr_text_enhanced or "", img.scene_label or "",
                    " ".join(_user_tags(img))
                ])).lower()
                if not meaningful:
                    continue
                # Apply contradiction filter FIRST
                if not _kw_ok(caption_text):
                    continue
                # Compound word matching: "ironman" counts as matched if both
                # "iron" AND "man" are present (even as separate words)
                def _word_in_haystack(w, hs):
                    if w in hs:
                        return True
                    parts = _COMPOUND_SPLITS.get(w)
                    if parts and all(p in hs for p in parts):
                        return True
                    return False
                # For 1-2 meaningful words: ALL must match
                # For 3+: majority (≥70%) must match
                matched = sum(1 for w in meaningful if _word_in_haystack(w, haystack))
                threshold = len(meaningful) if len(meaningful) <= 2 else max(2, len(meaningful) * 0.70)
                if matched >= threshold:
                    results.append({
                        "id": img.id, "filename": _img_url(img.filename), "score": 32.0,
                        "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                        "person_count": img.person_count or 0,
                        "caption_short": img.caption_short or "",
                        "caption_detailed": img.caption_detailed or "",
                        "quality_level": img.quality_level or "",
                        "quality_score": img.quality_score or 0,
                        "dominant_emotion": img.dominant_emotion or "",
                        "face_emotion_count": img.face_emotion_count or 0,
                        "aesthetic_score": img.aesthetic_score or 0,
                        "aesthetic_rating": img.aesthetic_rating or "",
                        "is_favorite": bool(img.is_favorite),
                        "user_tags": _user_tags(img),
                        "photo_note": getattr(img, "photo_note", "") or "",
                        "scene_label": img.scene_label or "",
                        "width": img.width, "height": img.height,
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

        elif sig_words and skip_kw_fallback:
            # Person search mode: drop any CLIP results that aren't person matches
            # (e.g. "sadie sink" → remove literal sink images from CLIP)
            person_img_ids = {r["id"] for r in results if r.get("matched_person")}
            if person_img_ids:
                results = [r for r in results if r.get("matched_person") or r["id"] in person_img_ids]
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
        if not results:
            # If CLIP found nothing but the tag fast-path did, return tag results directly
            if _tag_fast_results:
                logger.info(f"🏷️ CLIP found nothing — returning {len(_tag_fast_results)} tag-matched results")
                return {
                    "status": "found", "query": query,
                    "count": len(_tag_fast_results),
                    "results": _tag_fast_results[:top_k],
                }
            # Return people suggestions so user can identify who to search
            people_suggestions = []
            if name_candidates:
                all_people = db.query(Person).all()
                for p in all_people:
                    face_recs = db.query(DBFace).filter(DBFace.person_id == p.id).limit(1).all()
                    if face_recs and face_recs[0].image_id:
                        cover_img = db.query(DBImage).filter(DBImage.id == face_recs[0].image_id).first()
                        if cover_img:
                            people_suggestions.append({
                                "id": p.id,
                                "name": p.name,
                                "cover": _img_url(cover_img.filename)
                            })
            return {
                "status": "not_found",
                "message": f"No images matched '{query}'.",
                "people_suggestions": people_suggestions[:12],
            }
        # ── Merge tag fast-path results (score=97) before dedup ─────────────
        # Tag-matched images that weren't already in CLIP results get prepended
        # at the top so they always appear first regardless of CLIP score.
        if _tag_fast_results:
            existing_ids = {r["id"] for r in results}
            for _tr in _tag_fast_results:
                if _tr["id"] not in existing_ids:
                    results.insert(0, _tr)
                    existing_ids.add(_tr["id"])
                else:
                    # Already in results — boost its score to 97
                    for _r in results:
                        if _r["id"] == _tr["id"]:
                            _r["score"] = max(_r.get("score", 0), 97.0)
                            break
            results.sort(key=lambda x: x["score"], reverse=True)

        # ── BUGFIX: Final dedup by filename ─────────────────────────────────
        # Catches re-indexed duplicate DB rows (different id, same physical file)
        # that bypass the earlier id-based dedup in _score_candidates.
        _seen_fn_final  = set()
        _seen_cap_final = set()   # also dedup by caption — catches same-content re-uploads
        _deduped_final  = []
        for _r in results:
            _fn  = _r.get("filename") or ""
            _ct  = (_r.get("caption_short") or "").strip().lower()
            if _fn and _fn in _seen_fn_final:
                continue
            # Caption dedup: skip if identical caption already in results
            # (keep first = highest scored; favourites get boosted by tag merge so appear first)
            if _ct and _ct in _seen_cap_final:
                continue
            if _fn:
                _seen_fn_final.add(_fn)
            if _ct:
                _seen_cap_final.add(_ct)
            _deduped_final.append(_r)
        results = _deduped_final

        return {"status": "found", "query": query, "count": len(results), "results": results}
    finally:
        db.close()

@app.post("/search/describe")
def search_by_description(description: str = Form(...), top_k: int = Form(20)):
    if not description or not description.strip():
        return {"status": "error", "message": "Description empty"}
    if search_engine.index is None:
        return {"status": "error", "message": "No images indexed."}
    desc = _clean_query(description.strip())
    prompts = [desc, f"a photo of {desc}", f"an image showing {desc}", f"a picture of {desc}", f"{desc} photograph"]
    embs = [e for p in prompts for e in [search_engine.get_text_embedding(p, use_prompt_ensemble=False)] if e is not None]
    if not embs:
        return {"status": "error", "message": "Could not encode description."}
    query_emb = np.mean(embs, axis=0).astype("float32")
    norm = np.linalg.norm(query_emb)
    if norm > 1e-8:
        query_emb /= norm
    sig_words    = [w for w in desc.lower().split() if len(w) > 2]
    query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in desc.lower()]
    db = SessionLocal()
    try:
        results = _score_candidates(query_emb, sig_words, query_colors, db, top_k)
        if not results:
            return {"status": "not_found", "message": f"No images matched description '{desc}'."}
        return {"status": "found", "query": desc, "count": len(results), "results": results}
    finally:
        db.close()

@app.post("/search/hybrid")
async def search_hybrid(query: str = Form(""), file: UploadFile = File(None), text_weight: float = Form(0.6), image_weight: float = Form(0.4), top_k: int = Form(20)):
    if search_engine.index is None:
        return {"status": "error", "message": "No images indexed."}
    has_text  = query.strip() != ""
    has_image = file is not None and file.filename
    if not has_text and not has_image:
        return {"status": "error", "message": "Provide at least a query or an image."}
    if has_text and has_image:
        total_w = text_weight + image_weight
        if total_w <= 0:
            text_weight, image_weight = 0.6, 0.4
            total_w = 1.0
        text_weight /= total_w; image_weight /= total_w
    elif has_text:
        text_weight, image_weight = 1.0, 0.0
    else:
        text_weight, image_weight = 0.0, 1.0
    text_emb = None
    if has_text:
        text_emb = search_engine.get_text_embedding(_clean_query(query), use_prompt_ensemble=True)
    img_emb = None
    if has_image:
        import tempfile
        ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
        try:
            img_emb = search_engine.get_image_embedding(tmp_path)
        finally:
            try: os.remove(tmp_path)
            except Exception: pass
    if text_emb is not None and img_emb is not None:
        query_emb, extra_emb = text_emb, img_emb
    elif text_emb is not None:
        query_emb, extra_emb = text_emb, None
    elif img_emb is not None:
        query_emb, extra_emb = img_emb, None; text_weight, image_weight = 1.0, 0.0
    else:
        return {"status": "error", "message": "Could not encode query."}
    sig_words    = [w for w in query.lower().split() if len(w) > 2]
    query_colors = [rgb for name, rgb in COLOR_SCORE_MAP.items() if name in query.lower()]
    db = SessionLocal()
    try:
        results = _score_candidates(query_emb, sig_words, query_colors, db, top_k, extra_emb=extra_emb, text_weight=text_weight, image_weight=image_weight)
        if not results:
            return {"status": "not_found", "message": "No images matched."}
        return {"status": "found", "query": query, "count": len(results), "text_weight": text_weight, "image_weight": image_weight, "results": results}
    finally:
        db.close()

@app.post("/search/voice")
def voice_search_legacy(duration: int = Form(5)):
    try:
        transcribed = voice_engine.listen_and_transcribe(duration=duration)
        if not transcribed or not transcribed.strip():
            return {"status": "error", "message": "Could not hear anything."}
        result = search(query=transcribed.strip(), top_k=20)
        result["transcribed"] = transcribed.strip()
        return result
    except Exception as e:
        return {"status": "error", "message": f"Voice search failed: {str(e)}"}

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(20)):
    if search_engine.index is None:
        return {"status": "error", "message": "No images indexed."}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported format.")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)
    try:
        query_emb = search_engine.get_image_embedding(tmp_path)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
    if query_emb is None:
        return {"status": "error", "message": "Could not process image."}
    total = search_engine.index.ntotal
    q = query_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    distances, indices = search_engine.index.search(q, min(top_k * 3, total))
    db = SessionLocal()
    try:
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            score = float(dist)
            if score < 0.45: break
            img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
            if not img or img.is_trashed: continue
            results.append({"id": img.id, "filename": _img_url(img.filename), "score": round(score * 100, 2), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None, "person_count": img.person_count or 0})
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        if not results:
            return {"status": "not_found", "message": "No visually similar images found."}
        return {"status": "found", "count": len(results), "results": results}
    finally:
        db.close()

@app.post("/search/color")
def search_by_color(color: str = Form(...), top_k: int = Form(20)):
    """
    Color search using HSV hue matching instead of raw RGB distance.
    Much more accurate — a blue sky photo scores high for 'blue' regardless
    of how many green trees are also in the frame.
    """
    import colorsys

    # ── Hue ranges in degrees (0-360) for each color ────────────────────────
    # Each entry: (hue_center, hue_half_width, min_saturation, min_value)
    COLOR_HSV = {
        "red":    [(  0, 18, 0.35, 0.20), (360, 18, 0.35, 0.20)],  # red wraps around 0/360
        "orange": [( 25, 15, 0.45, 0.25)],
        "yellow": [( 55, 18, 0.40, 0.30)],
        "green":  [(120, 40, 0.30, 0.15)],
        "blue":   [(220, 45, 0.30, 0.15)],
        "purple": [(280, 30, 0.25, 0.15)],
        "pink":   [(330, 25, 0.25, 0.30)],
        "white":  None,   # special: high value, low saturation
        "black":  None,   # special: low value
        "gray":   None,   # special: low saturation
        "grey":   None,
        "brown":  [( 25, 18, 0.35, 0.15)],  # like orange but lower value
    }

    color_key = color.strip().lower()
    if color_key not in COLOR_HSV:
        color_key = next((k for k in COLOR_HSV if k in color_key), None)
    if not color_key:
        return {"status": "error", "message": f"Unknown color '{color}'."}

    def _color_score(r, g, b, ckey):
        """Score 0-100 how well an RGB pixel matches the target color."""
        r_, g_, b_ = (r or 0)/255.0, (g or 0)/255.0, (b or 0)/255.0
        h, s, v = colorsys.rgb_to_hsv(r_, g_, b_)
        hue_deg = h * 360.0

        if ckey in ("white", "grey", "gray"):
            # High brightness, low saturation
            if ckey == "white":
                return max(0.0, (v - 0.75) / 0.25 * 100) * max(0.0, (0.25 - s) / 0.25)
            else:  # gray
                sat_score = max(0.0, (0.20 - s) / 0.20)
                val_score = max(0.0, 1.0 - abs(v - 0.50) / 0.50)
                return sat_score * val_score * 100
        if ckey == "black":
            return max(0.0, (0.25 - v) / 0.25 * 100)

        ranges = COLOR_HSV[ckey]
        if not ranges:
            return 0.0

        best = 0.0
        for entry in ranges:
            hc, hw, min_s, min_v = entry
            # Hue distance (circular)
            diff = abs(hue_deg - hc)
            if diff > 180: diff = 360 - diff
            if diff > hw * 2:
                continue
            hue_score  = max(0.0, 1.0 - diff / (hw * 2))
            sat_score  = min(1.0, max(0.0, (s - min_s) / (1.0 - min_s + 0.01)))
            val_score  = min(1.0, max(0.0, (v - min_v) / (1.0 - min_v + 0.01)))
            # brown special case: penalise high-value (bright) oranges
            if ckey == "brown":
                val_score *= max(0.0, 1.0 - max(0.0, v - 0.55) / 0.45)
            score = hue_score * sat_score * val_score
            best = max(best, score)
        return round(best * 100, 2)

    db = SessionLocal()
    try:
        images = _live(db).filter(
            DBImage.avg_r != None, DBImage.avg_g != None, DBImage.avg_b != None
        ).all()
        if not images:
            return {"status": "not_found", "message": "No color data. Upload photos first."}

        scored = []
        for img in images:
            s = _color_score(img.avg_r, img.avg_g, img.avg_b, color_key)
            if s >= 10.0:  # low threshold — HSV matching is already precise
                scored.append((s, img))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_k]

        if not scored:
            return {"status": "not_found", "message": f"No images matched color '{color}'."}
        return {
            "status": "found", "query": color, "count": len(scored),
            "results": [{
                "id": img.id, "filename": _img_url(img.filename),
                "score": round(s, 2),
                "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                "caption_short": img.caption_short or "",
                "user_tags": _user_tags(img),
                "dominant_emotion": img.dominant_emotion,
                "person_count": img.person_count or 0,
            } for s, img in scored]
        }
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# REPROCESS EMOTIONS (fixes existing photos that have "neutral" default)
# ────────────────────────────────────────────────────────────────────────────
@app.post("/reprocess-colors")
def reprocess_colors():
    """Recompute dominant color for all existing photos using weighted histogram."""
    import colorsys as _cs2
    from PIL import Image as PILImage2
    db = SessionLocal()
    try:
        images = _live(db).all()
        done = 0
        for img in images:
            fpath = img.original_path or os.path.join(IMAGE_DIR, img.filename)
            if not os.path.exists(fpath):
                continue
            try:
                pil = PILImage2.open(fpath).convert("RGB").resize((64,64))
                arr2 = np.array(pil, dtype=np.float32)
                r_v = arr2[:,:,0].flatten(); g_v = arr2[:,:,1].flatten(); b_v = arr2[:,:,2].flatten()
                wts = np.array([
                    _cs2.rgb_to_hsv(r/255,g/255,b/255)[1] * _cs2.rgb_to_hsv(r/255,g/255,b/255)[2] + 0.01
                    for r,g,b in zip(r_v,g_v,b_v)
                ])
                tw = wts.sum()
                img.avg_r = float(np.dot(r_v, wts)/tw)
                img.avg_g = float(np.dot(g_v, wts)/tw)
                img.avg_b = float(np.dot(b_v, wts)/tw)
                done += 1
            except Exception as e:
                logger.warning(f"Color reprocess failed {img.id}: {e}")
        db.commit()
        return {"status": "done", "updated": done}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@app.post("/reprocess-emotions")
async def reprocess_emotions(background_tasks: BackgroundTasks, limit: int = 0):
    """
    Re-detect emotions on all uploaded images that still have the default
    'neutral' label or no emotion set. Runs in background so the API returns
    immediately.  limit=0 means process ALL.
    """
    db = SessionLocal()
    try:
        q = _live(db)
        if limit > 0:
            q = q.limit(limit)
        image_ids = [
            (img.id, img.original_path or os.path.join(IMAGE_DIR, img.filename))
            for img in q.all()
        ]
    finally:
        db.close()

    def _run_reprocess(pairs):
        done = 0
        for img_id, fpath in pairs:
            if not os.path.exists(fpath):
                continue
            try:
                # ── GATE: use InsightFace face count, not Faster R-CNN person_count ──
                # person_count = 0 for close-up portraits (no full body visible).
                # InsightFace accurately detects human faces regardless of body visibility.
                # Cat faces, car headlights, turtles → InsightFace gives 0 → skip model.
                _db_chk = SessionLocal()
                try:
                    _face_count = _db_chk.query(DBFace).filter(DBFace.image_id == img_id).count()
                except Exception:
                    _face_count = -1  # unknown → let it through
                finally:
                    _db_chk.close()

                if _face_count == 0:
                    # No real human faces → clear any previously wrong emotion label
                    _db_n = SessionLocal()
                    try:
                        _db_n.query(DBImage).filter(DBImage.id == img_id).update({
                            "dominant_emotion": "neutral", "face_emotion_count": 0,
                            "emotion_data": "[]",
                        }, synchronize_session=False)
                        _db_n.commit()
                    finally:
                        _db_n.close()
                    continue

                ed = emotion_detection.detect_emotions(fpath)
                # Confidence filter — same as _enrich_image
                ed = [e for e in ed if e.get("confidence", 0) >= 0.50]
                dominant = ed[0]["emotion"] if ed else "neutral"
                db2 = SessionLocal()
                try:
                    db2.query(DBImage).filter(DBImage.id == img_id).update({
                        "dominant_emotion":   dominant,
                        "face_emotion_count": len(ed),
                        "emotion_data":       _json.dumps(ed),
                    }, synchronize_session=False)
                    db2.commit()
                    done += 1
                finally:
                    db2.close()
            except Exception as e:
                logger.warning(f"Emotion reprocess failed for {img_id}: {e}")
        logger.info(f"✅ Emotion reprocess complete: {done}/{len(pairs)} images updated")

    background_tasks.add_task(_run_reprocess, image_ids)
    return {
        "status": "started",
        "message": f"Reprocessing emotions for {len(image_ids)} images in background",
        "count": len(image_ids)
    }


@app.post("/reprocess-names")
def reprocess_names():
    """
    Re-run auto-naming on ALL existing person clusters.
    Resets garbage names (ALL-CAPS brands, OCR fragments, short initials)
    back to "Person N", then re-runs the improved auto-naming logic.
    """
    db = SessionLocal()
    try:
        # ── Step 1: Reset bad auto-assigned names ───────────────────────────
        people_all = db.query(Person).all()
        reset_count = 0
        for i, p in enumerate(people_all):
            pn = (p.name or "").strip()
            # Never touch manually typed names (user renamed them — they know best)
            # We can't tell manual from auto, so use heuristics to detect garbage:
            should_reset = False

            if not pn or len(pn) < 2:
                should_reset = True
            elif not any(c.isalpha() for c in pn):
                should_reset = True
            else:
                name_words = pn.split()
                # Check every word in the name
                bad_word_count = 0
                for nw in name_words:
                    nw_clean = nw.rstrip(".,!?;:")
                    # ALL-CAPS word (brand name / jersey text)
                    if nw_clean.isalpha() and nw_clean == nw_clean.upper() and len(nw_clean) > 2:
                        bad_word_count += 1
                    # Too short (< 3 chars = initials / OCR fragment)
                    elif len(nw_clean) < 3:
                        bad_word_count += 1
                    # Known garbage word
                    elif nw_clean.lower() in _BLIP_NON_NAMES:
                        bad_word_count += 1
                    # No vowels (pure consonant = OCR garbage)
                    elif not any(c in "aeiouAEIOU" for c in nw_clean):
                        bad_word_count += 1
                # If ANY word in the name is garbage → reset the whole name
                if bad_word_count > 0:
                    should_reset = True

            if should_reset:
                p.name = f"Person {i + 1}"
                reset_count += 1

        db.commit()
        logger.info(f"🔄 Reset {reset_count} bad auto-names back to defaults")

        # ── Step 2: Re-run auto-naming with improved logic ──────────────────
        people_all = db.query(Person).all()
        person_map = {i: p.id for i, p in enumerate(people_all)}
        _auto_name_people(db, person_map)
        db.commit()

        named = db.query(Person).filter(~Person.name.startswith("Person ")).count()
        total = db.query(Person).count()
        return {
            "status": "done",
            "reset": reset_count,
            "named": named,
            "total": total,
            "message": f"{named}/{total} people named · {reset_count} bad names reset"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# FACES / PEOPLE
# ────────────────────────────────────────────────────────────────────────────
@app.get("/faces")
def get_faces(person_id: int = Query(None)):
    db = SessionLocal()
    try:
        if person_id:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise HTTPException(status_code=404)
            rows = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == person_id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
            cover  = _img_url(rows[0].filename) if rows else None
            images = [{"id": img.id, "filename": _img_url(img.filename), "thumbnail": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None} for img in rows]
            return {"id": person.id, "name": person.name, "face_count": len(images), "cover": cover, "images": images, "results": images}
        else:
            people  = db.query(Person).all()
            results = []
            for p in people:
                imgs = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == p.id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
                if not imgs: continue
                results.append({"id": p.id, "name": p.name, "count": len(imgs), "cover": _img_url(imgs[0].filename)})
            return {"results": results, "count": len(results)}
    finally:
        db.close()

@app.get("/people/search")
def search_people_by_name(q: str = Query(...)):
    """Search people by name — returns matching persons with their photos."""
    db = SessionLocal()
    try:
        q_lower = q.lower().strip()
        if not q_lower:
            return {"results": []}
        all_people = db.query(Person).all()
        matched = []
        for p in all_people:
            pn = (p.name or "").lower().strip()
            if not pn or pn in ("unknown",):
                continue
            parts = pn.split()
            if q_lower in pn or any(q_lower in part for part in parts) or any(part in q_lower for part in parts if len(part)>=2):
                face_records = db.query(DBFace).filter(DBFace.person_id == p.id).all()
                img_ids = list({f.image_id for f in face_records if f.image_id})
                imgs = _live(db).filter(DBImage.id.in_(img_ids)).order_by(DBImage.timestamp.desc()).all() if img_ids else []
                cover = _img_url(imgs[0].filename) if imgs else None
                matched.append({
                    "id": p.id, "name": p.name, "count": len(imgs), "cover": cover,
                    "results": [{"id": img.id, "filename": _img_url(img.filename),
                                 "score": 92.0, "matched_person": p.name,
                                 "caption_short": img.caption_short or "",
                                 "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                                 "user_tags": _user_tags(img),
                                 "person_count": img.person_count or 0} for img in imgs]
                })
        return {"query": q, "count": len(matched), "results": matched}
    finally:
        db.close()


@app.get("/people/{person_id}")
def get_person(person_id: int):
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=404)
        rows = db.query(DBImage).join(DBFace, DBFace.image_id == DBImage.id).filter(DBFace.person_id == person_id).filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None)).filter(DBImage.person_count > 0).distinct().all()
        cover  = _img_url(rows[0].filename) if rows else None
        images = [{"id": img.id, "filename": _img_url(img.filename), "thumbnail": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None} for img in rows]
        return {"id": person.id, "name": person.name, "face_count": len(images), "cover": cover, "images": images, "results": images}
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
    return {"status": "no_match"}


# ────────────────────────────────────────────────────────────────────────────
# ALBUMS
# ────────────────────────────────────────────────────────────────────────────
@app.get("/albums/{album_id}")
def get_album_by_id(album_id: int):
    db = SessionLocal()
    try:
        album = db.query(Album).filter(Album.id == album_id).first()
        if not album:
            raise HTTPException(status_code=404, detail=f"Album {album_id} not found")
        images = _live(db).filter(DBImage.album_id == album_id).order_by(DBImage.timestamp).all()
        cover  = _img_url(images[0].filename) if images else None
        date_str = ""
        if album.start_date:
            date_str = album.start_date.strftime("%b %Y")
            if album.end_date and album.end_date.month != album.start_date.month:
                date_str += f" – {album.end_date.strftime('%b %Y')}"
        img_list = [{"id": img.id, "filename": _img_url(img.filename), "date": img.timestamp.isoformat() if img.timestamp else None, "caption_short": img.caption_short, "ocr_text_enhanced": img.ocr_text_enhanced, "quality_score": img.quality_score, "quality_level": img.quality_level, "dominant_emotion": img.dominant_emotion, "aesthetic_score": img.aesthetic_score} for img in images]
        return {"id": album.id, "title": album.title, "type": album.type, "description": album.description, "date": date_str, "cover": cover, "start_date": album.start_date.isoformat() if album.start_date else None, "end_date": album.end_date.isoformat() if album.end_date else None, "image_count": len(images), "images": img_list, "results": img_list, "thumbnails": [_img_url(img.filename) for img in images[:4]]}
    finally:
        db.close()

@app.get("/albums")
def get_albums(album_id: int = Query(None)):
    db = SessionLocal()
    try:
        if album_id:
            return get_album_by_id(album_id)
        albums  = db.query(Album).all()
        results = []
        for a in albums:
            album_images = _live(db).filter(DBImage.album_id == a.id).all()
            # Show empty manual albums — user just created them
            if not album_images and a.type == "event":
                continue  # hide empty auto-event albums, show empty manual ones
            date_str = ""
            if a.start_date:
                date_str = a.start_date.strftime("%b %Y")
                if a.end_date and a.end_date.month != a.start_date.month:
                    date_str += f" – {a.end_date.strftime('%b %Y')}"
            cover = _img_url(album_images[0].filename) if album_images else None
            results.append({"id": a.id, "title": a.title, "type": a.type or "manual",
                            "description": a.description, "date": date_str,
                            "cover": cover, "count": len(album_images),
                            "thumbnails": [_img_url(img.filename) for img in album_images[:4]]})
        return {"results": results, "count": len(results)}
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# ALBUM CRUD
# ────────────────────────────────────────────────────────────────────────────

@app.post("/albums/create")
def create_album(
    title: str = Form(...),
    description: str = Form(""),
    image_ids: str = Form(""),   # comma-separated image IDs to add immediately
):
    """Create a new manual album, optionally with a set of images."""
    db = SessionLocal()
    try:
        album = Album(
            title=title.strip(),
            description=description.strip() or None,
            type="manual",
        )
        db.add(album); db.flush()

        added = 0
        if image_ids.strip():
            ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
            for img in _live(db).filter(DBImage.id.in_(ids)).all():
                img.album_id = album.id
                added += 1
            # Set album dates from images
            imgs = _live(db).filter(DBImage.album_id == album.id).all()
            timestamps = [i.timestamp for i in imgs if i.timestamp]
            if timestamps:
                album.start_date = min(timestamps)
                album.end_date   = max(timestamps)

        db.commit()
        logger.info(f"✅ Album created: '{title}' (id={album.id}, {added} images)")
        return {"status": "created", "id": album.id, "title": album.title, "image_count": added}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@app.delete("/albums/{album_id}/delete")
def delete_album(album_id: int):
    """Delete an album (photos are kept, just unlinked from album)."""
    db = SessionLocal()
    try:
        album = db.query(Album).filter(Album.id == album_id).first()
        if not album:
            raise HTTPException(404, "Album not found")
        db.query(DBImage).filter(DBImage.album_id == album_id).update({"album_id": None})
        db.delete(album); db.commit()
        return {"status": "deleted", "id": album_id}
    except HTTPException: raise
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@app.delete("/albums/empty/cleanup")
def cleanup_empty_albums():
    """Delete all empty manual albums (useful after accidental duplicates)."""
    db = SessionLocal()
    try:
        manual_albums = db.query(Album).filter(Album.type == "manual").all()
        deleted = []
        for a in manual_albums:
            count = db.query(DBImage).filter(DBImage.album_id == a.id).count()
            if count == 0:
                deleted.append(a.title)
                db.delete(a)
        db.commit()
        return {"status": "done", "deleted": len(deleted), "titles": deleted}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@app.post("/albums/{album_id}/rename")
def rename_album(album_id: int, title: str = Form(...), description: str = Form("")):
    """Rename an album and optionally update its description."""
    db = SessionLocal()
    try:
        album = db.query(Album).filter(Album.id == album_id).first()
        if not album:
            raise HTTPException(404, "Album not found")
        album.title = title.strip()
        if description.strip():
            album.description = description.strip()
        db.commit()
        return {"status": "renamed", "id": album_id, "title": album.title}
    except HTTPException: raise
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@app.post("/albums/{album_id}/add-images")
def add_images_to_album(album_id: int, image_ids: str = Form(...)):
    """Add a set of images (comma-separated IDs) to an existing album."""
    db = SessionLocal()
    try:
        album = db.query(Album).filter(Album.id == album_id).first()
        if not album:
            raise HTTPException(404, "Album not found")
        ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
        updated = _live(db).filter(DBImage.id.in_(ids)).update(
            {"album_id": album_id}, synchronize_session=False
        )
        # Refresh album dates
        imgs = _live(db).filter(DBImage.album_id == album_id).all()
        timestamps = [i.timestamp for i in imgs if i.timestamp]
        if timestamps:
            album.start_date = min(timestamps)
            album.end_date   = max(timestamps)
        db.commit()
        return {"status": "added", "album_id": album_id, "added": updated}
    except HTTPException: raise
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@app.post("/albums/{album_id}/remove-images")
def remove_images_from_album(album_id: int, image_ids: str = Form(...)):
    """Remove images from an album (images are kept, just unlinked)."""
    db = SessionLocal()
    try:
        ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
        _live(db).filter(DBImage.id.in_(ids), DBImage.album_id == album_id).update(
            {"album_id": None}, synchronize_session=False
        )
        db.commit()
        return {"status": "removed", "album_id": album_id}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# DUPLICATES / EXPLORE / STATS
# ────────────────────────────────────────────────────────────────────────────
@app.get("/duplicates")
def get_duplicates():
    db = SessionLocal()
    try:
        all_images = _live(db).all()
        if not all_images:
            return {"status": "found", "duplicate_groups": [], "total_groups": 0}
        groups    = duplicate_engine.find_duplicates_fast(all_images, hamming_threshold=5)
        formatted = [{"count": g["count"], "total_size": g["total_size"], "images": [{"id": img["id"], "filename": _img_url(img["filename"]), "thumbnail": _img_url(img["filename"]), "size": img["size"]} for img in g["images"]]} for g in groups]
        return {"status": "found", "duplicate_groups": formatted, "total_groups": len(formatted)}
    except Exception as e:
        logger.error(f"Duplicates error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "duplicate_groups": [], "total_groups": 0}
    finally:
        db.close()

@app.get("/explore/random")
def explore_random(count: int = Query(12)):
    import random as _random
    db = SessionLocal()
    try:
        all_ids = [row[0] for row in _live(db).with_entities(DBImage.id).all()]
        if not all_ids:
            return {"status": "not_found", "results": []}
        sampled = _random.sample(all_ids, min(count, len(all_ids)))
        imgs    = db.query(DBImage).filter(DBImage.id.in_(sampled)).all()
        return {"status": "found", "count": len(imgs), "results": [{"id": img.id, "filename": _img_url(img.filename), "timestamp": img.timestamp.isoformat() if img.timestamp else None, "location": {"lat": img.lat, "lon": img.lon} if img.lat and img.lon else None, "person_count": img.person_count or 0} for img in imgs]}
    finally:
        db.close()

@app.get("/stats")
def get_stats():
    import collections
    db = SessionLocal()
    try:
        total_images    = db.query(DBImage).count()
        total_faces     = db.query(DBFace).count()
        total_people    = db.query(Person).count()
        total_albums    = db.query(Album).count()
        total_favorites = _live(db).filter(DBImage.is_favorite == True).count()
        total_trashed   = db.query(DBImage).filter(DBImage.is_trashed == True).count()
        indexed         = search_engine.index.ntotal if search_engine.index else 0
        # AI object-detection labels (deduplicated per image)
        ai_counter = collections.Counter()
        for row in db.query(DBImage.scene_label).filter(DBImage.scene_label != None).all():
            seen_in_image = set()
            for tag in (row[0] or "").split(","):
                t = tag.strip().lower()
                if t and t not in seen_in_image:
                    ai_counter[t] += 1
                    seen_in_image.add(t)
        top_tags = [{"tag": t, "count": c} for t, c in ai_counter.most_common(10)]

        # User-defined tags
        user_tag_counter = collections.Counter()
        for row in db.execute(text("SELECT user_tags FROM images WHERE user_tags IS NOT NULL AND user_tags != '[]' AND (is_trashed IS NULL OR is_trashed=0)")):
            try:
                import json as _j
                for t in _j.loads(row[0]): user_tag_counter[t] += 1
            except Exception: pass
        top_user_tags = [{"tag": t, "count": c} for t, c in user_tag_counter.most_common(20)]
        COLOR_BUCKETS = {"red": (220,50,50), "orange": (230,120,40), "yellow": (220,210,50), "green": (50,160,50), "blue": (50,80,220), "purple": (120,50,180), "pink": (230,130,160), "white": (240,240,240), "black": (20,20,20), "gray": (128,128,128), "brown": (140,90,50)}
        color_dist = {k: 0 for k in COLOR_BUCKETS}
        for img in _live(db).filter(DBImage.avg_r != None).all():
            img_rgb = np.array([img.avg_r or 0, img.avg_g or 0, img.avg_b or 0], dtype=np.float32)
            best_bucket = min(COLOR_BUCKETS, key=lambda k: np.linalg.norm(img_rgb - np.array(COLOR_BUCKETS[k], dtype=np.float32)))
            color_dist[best_bucket] += 1
        return {"total_images": total_images, "total_faces": total_faces, "total_people": total_people, "total_albums": total_albums, "total_favorites": total_favorites, "total_trashed": total_trashed, "indexed_vectors": indexed, "top_tags": top_tags, "top_user_tags": top_user_tags, "color_distribution": sorted([{"color": k, "count": v} for k, v in color_dist.items() if v > 0], key=lambda x: x["count"], reverse=True)}
    finally:
        db.close()


# ── Common non-name words returned by BLIP when it doesn't know the name ───
_BLIP_NON_NAMES = {
    # articles / pronouns / common words
    "a", "an", "the", "it", "he", "she", "they", "him", "her", "them",
    "his", "hers", "their", "we", "our", "you", "your", "my", "i",
    # question words (critical — BLIP echoes these back)
    "what", "who", "which", "where", "when", "how", "why", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "has",
    "have", "had", "will", "would", "could", "should", "may", "might",
    "can", "name", "named", "called", "known",
    # generic person descriptions
    "man", "woman", "person", "people", "boy", "girl", "child", "adult",
    "human", "face", "faces", "subject", "individual", "individuals",
    "unknown", "someone", "somebody", "anyone", "nobody", "one",
    # job titles / roles (BLIP often returns these)
    "celebrity", "actor", "actress", "model", "player", "star", "singer",
    "politician", "athlete", "musician", "artist", "president", "director",
    "manager", "leader", "officer", "official", "member", "representative",
    # visual/image words (BLIP captions)
    "image", "photo", "picture", "figure", "portrait", "photograph",
    "this", "that", "these", "those", "here", "there",
    # caption filler words BLIP uses
    "main", "shows", "show", "showing", "appears", "appear", "looking",
    "wearing", "holding", "standing", "sitting", "smiling",
    "in", "on", "at", "of", "for", "with", "from", "by", "to",
    "and", "or", "but", "not", "also", "as", "so", "then", "than",
    "black", "white", "two", "three", "four", "five", "several", "many",
    # Sports context — appear on jerseys, banners, scoreboards
    "india", "team", "cup", "match", "captain", "wicket", "wizard",
    "champion", "cricket", "football", "sport", "league", "series",
    "trophy", "test", "ipl", "bcci", "odi", "t20", "final", "world",
    "semi", "quarter", "squad", "player", "coach", "umpire", "referee",
    # Brand/sponsor words (F1, cricket, general)
    "petronas", "stake", "marlboro", "ferrari", "mercedes", "mclaren",
    "alpine", "williams", "haas", "redbull", "birla", "estates", "bank",
    "mutual", "fund", "limited", "pvt", "company", "corp", "inc", "ltd",
    "group", "holding", "sponsor", "official", "partner",
    # Common OCR fragments from Indian sports context
    "ndia", "iindia", "ldcup", "ineqs", "eironas", "hak", "zon",
    # Titles / honorifics misread as names
    "master", "mister", "miss", "mrs", "sir", "dr", "prof", "mr",
    # Generic English words that look capitalised in headlines
    "new", "old", "big", "great", "good", "best", "top", "live",
    "real", "true", "blue", "gold", "king", "queen", "super", "ultra",
}

# ── Patterns that definitively mark a word as NOT a person name ─────────────
def _is_valid_name_word(w: str, allow_short: bool = False) -> bool:
    """
    Return True only if `w` looks like part of a real person's name.
    Rejects: ALL-CAPS brands, too-short initials, non-alpha, known non-names.
    """
    w_clean = w.rstrip(".,!?;:'\"")
    if not w_clean:
        return False
    # Must be purely alphabetic (no digits, no hyphens, no underscores)
    if not w_clean.isalpha():
        return False
    # ALL-CAPS words are brand names / acronyms / jersey text, not person names
    if w_clean == w_clean.upper() and len(w_clean) > 2:
        return False
    # Minimum length — real given names are rarely < 3 chars in isolation
    min_len = 3 if allow_short else 4
    if len(w_clean) < min_len:
        return False
    # Must start with uppercase (proper noun)
    if not w_clean[0].isupper():
        return False
    # Must not be a known non-name
    if w_clean.lower() in _BLIP_NON_NAMES:
        return False
    # Must contain at least one vowel (pure consonant strings = OCR garbage)
    if not any(c in "aeiouAEIOU" for c in w_clean):
        return False
    return True


def _extract_name_from_vqa(vqa_text: str) -> str:
    """
    Parse a BLIP VQA answer to extract a plausible person name.
    Very strict — only returns something if it looks like a real proper name.
    Returns empty string if uncertain.
    """
    if not vqa_text:
        return ""
    text = vqa_text.strip()

    # Strip common prefixes BLIP adds
    for prefix in [
        "the person is ", "this is ", "it is ", "that is ",
        "his name is ", "her name is ", "the name is ",
        "the man is ", "the woman is ", "i think it is ",
        "i think it's ", "i believe it is ", "appears to be ",
        "looks like ", "this appears to be ", "the answer is ",
    ]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):]

    # Take just the first 1-2 words
    words = text.split()[:2]
    if not words:
        return ""

    # Validate each word — allow short (3-char) first names only in 2-word combos
    if len(words) == 2:
        w1_ok = _is_valid_name_word(words[0], allow_short=True)
        w2_ok = _is_valid_name_word(words[1], allow_short=False)
        if w1_ok and w2_ok:
            name = f"{words[0].rstrip('.,!?;:')} {words[1].rstrip('.,!?;:')}"
            return name if 4 <= len(name) <= 35 else ""
        if w1_ok:
            w = words[0].rstrip(".,!?;:")
            return w if 4 <= len(w) <= 35 else ""
        return ""
    else:
        w = words[0].rstrip(".,!?;:")
        if _is_valid_name_word(w, allow_short=False):
            return w if 4 <= len(w) <= 35 else ""
        return ""


def _auto_name_people(db, person_map: dict):
    """
    For each newly created person cluster, try to auto-name them.
    Strategy (strict):
      1. VQA "person" answers get highest weight (3 pts each) — most reliable
      2. OCR proper nouns that appear in MULTIPLE images of the same cluster get 1 pt each
         (single-image OCR hits are often jersey text / background banners → ignored)
      3. Only assign a name if the top candidate has >= 50% of photos voting for it
         AND if it's a VQA-sourced name, OR >= 2 images (to prevent single-image OCR garbage)
    """
    import json as _jj
    from collections import Counter

    for label, person_id in person_map.items():
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            continue
        # Only auto-name if still using the default label
        if not person.name.startswith("Person "):
            continue

        face_records = db.query(DBFace).filter(DBFace.person_id == person_id).all()
        img_ids = list({f.image_id for f in face_records if f.image_id})
        if not img_ids:
            continue

        images = db.query(DBImage).filter(DBImage.id.in_(img_ids)).all()
        n_images = len(images)

        # Separate counters: VQA (trusted) vs OCR (needs cross-image confirmation)
        vqa_counter = Counter()
        ocr_counter = Counter()   # counts how many IMAGES contain each OCR name

        for img in images:
            # ── VQA: most reliable, use directly ────────────────────────────
            if img.caption_vqa:
                try:
                    vqa = _jj.loads(img.caption_vqa)
                    vqa_name = _extract_name_from_vqa(vqa.get("person", ""))
                    if vqa_name:
                        vqa_counter[vqa_name] += 1
                except Exception:
                    pass

            # ── OCR: scan for proper-noun pairs and single words ─────────────
            # Only count a name ONCE per image (to prevent a single jersey from
            # winning just because the text appears 10 times in one photo).
            ocr = img.ocr_text_enhanced or ""
            if ocr:
                seen_in_this_image = set()
                words = ocr.split()
                for i, w in enumerate(words):
                    # Try two-word name first
                    if i + 1 < len(words):
                        w2 = words[i + 1].rstrip(".,!?;:'\"")
                        w1 = w.rstrip(".,!?;:'\"")
                        if (_is_valid_name_word(w1, allow_short=True) and
                                _is_valid_name_word(w2, allow_short=False)):
                            two_word = f"{w1} {w2}"
                            if two_word not in seen_in_this_image:
                                seen_in_this_image.add(two_word)
                                ocr_counter[two_word] += 1

                    # Single word fallback
                    w1 = w.rstrip(".,!?;:'\"")
                    if (_is_valid_name_word(w1, allow_short=False) and
                            w1 not in seen_in_this_image):
                        seen_in_this_image.add(w1)
                        ocr_counter[w1] += 1

        # ── Decision logic ────────────────────────────────────────────────────
        best_name = None
        best_score = 0

        # VQA wins if it appears in >= 25% of images (e.g. 1 out of 4)
        if vqa_counter:
            top_vqa, top_vqa_count = vqa_counter.most_common(1)[0]
            min_vqa = max(1, n_images * 0.25)
            if top_vqa_count >= min_vqa:
                best_name  = top_vqa
                best_score = top_vqa_count * 3  # weighted

        # OCR only wins if:
        # (a) NO VQA name found, AND
        # (b) name appears in >= 2 different images (cross-image confirmation), AND
        # (c) appears in >= 40% of the cluster's images
        if not best_name and ocr_counter:
            top_ocr, top_ocr_count = ocr_counter.most_common(1)[0]
            min_ocr_images = max(2, n_images * 0.40)
            if top_ocr_count >= min_ocr_images:
                best_name  = top_ocr
                best_score = top_ocr_count

        if best_name:
            person.name = best_name
            logger.info(f"🏷️  Auto-named Person {person_id} → '{best_name}' "
                        f"(score={best_score}, vqa={dict(vqa_counter.most_common(3))}, "
                        f"n_images={n_images})")
        else:
            top_vqa_str = str(vqa_counter.most_common(1)) if vqa_counter else "none"
            top_ocr_str = str(ocr_counter.most_common(3)) if ocr_counter else "none"
            logger.info(f"⚠️  No confident name for Person {person_id} "
                        f"(n_images={n_images}, vqa={top_vqa_str}, top_ocr={top_ocr_str})")


# ────────────────────────────────────────────────────────────────────────────
# RECLUSTER
# ────────────────────────────────────────────────────────────────────────────
@app.post("/recluster")

def recluster():
    db = SessionLocal()
    try:
        logger.info("🔄 Recluster: saving manually assigned names before reset…")

        # ── Save manually named people BEFORE wiping everything ─────────────
        # For each person with a non-default name, remember which face embeddings
        # (as indices into the face table) belonged to them.
        saved_names = {}  # face_id_set -> name
        for person in db.query(Person).all():
            pname = (person.name or "").strip()
            # Only keep explicitly renamed people (skip "Person N" defaults)
            if re.match(r"^person\s+\d+$", pname.lower()):
                continue
            if not pname or pname.lower() in ("unknown", ""):
                continue
            face_ids = frozenset(
                f.id for f in db.query(DBFace).filter(DBFace.person_id == person.id).all()
            )
            if face_ids:
                saved_names[face_ids] = pname
                logger.info(f"💾 Saved name '{pname}' ({len(face_ids)} faces)")

        logger.info(f"🔄 Recluster: clearing old assignments…")
        db.query(DBFace).update({"person_id": None})
        db.query(Person).delete()

        # ── CRITICAL: only clear album_id for EVENT album images ─────────────
        # db.query(DBImage).update({"album_id": None})  ← old code wiped MANUAL albums too
        # Manual albums (type="manual") must be preserved — user curated them.
        # Only event albums (type="event") are rebuilt automatically.
        event_album_ids = [
            a.id for a in db.query(Album).filter(Album.type == "event").all()
        ]
        if event_album_ids:
            db.query(DBImage).filter(
                DBImage.album_id.in_(event_album_ids)
            ).update({"album_id": None}, synchronize_session=False)

        db.query(Album).filter(Album.type == "event").delete()
        db.commit()

        # ── Load face records with image dimensions for size-filtering ────────
        # Background faces (tiny, far-away people) corrupt clusters — the same
        # main person ends up split into multiple clusters because their face
        # sometimes appears blurry/small in group shots.
        # Strategy: for each image, only cluster faces that are "prominent":
        #   - face area >= MIN_FACE_RATIO of image area (ignores tiny bystanders)
        #   - if multiple faces pass the threshold, keep top MAX_FACES_PER_IMAGE
        #     by size (supports group photos with 2-3 main subjects)
        MIN_FACE_RATIO   = 0.005   # face must cover >= 0.5% of image area
                                    # 1920×1080 image → face must be >= ~100×100 px
        MAX_FACES_PER_IMAGE = 4    # keep at most 4 largest faces per image

        # Build image dimension lookup: image_id → (width, height)
        img_dims = {}
        for img in db.query(DBImage.id, DBImage.width, DBImage.height).all():
            if img.width and img.height:
                img_dims[img.id] = (img.width, img.height)

        face_records            = db.query(DBFace).filter(DBFace.face_embedding != None).all()
        embeddings, valid_faces = [], []
        skipped = 0
        bg_filtered = 0

        # Group faces by image to allow per-image top-N selection
        from collections import defaultdict
        import json as _jfilt
        faces_by_image = defaultdict(list)
        for fr in face_records:
            if not fr.image_id:
                continue
            try:
                emb = np.frombuffer(fr.face_embedding, dtype=np.float32).copy()
                if emb.shape[0] != 512:
                    skipped += 1
                    continue
                # Parse bbox to get face area
                face_area_ratio = 1.0  # default: include if no bbox/dims available
                if fr.bbox and fr.image_id in img_dims:
                    try:
                        bbox = _jfilt.loads(fr.bbox)  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        face_area = max(0, x2 - x1) * max(0, y2 - y1)
                        img_w, img_h = img_dims[fr.image_id]
                        img_area = img_w * img_h
                        face_area_ratio = face_area / img_area if img_area > 0 else 1.0
                    except Exception:
                        face_area_ratio = 1.0  # parse failed → include by default
                faces_by_image[fr.image_id].append((face_area_ratio, fr, emb))
            except Exception as e:
                logger.warning(f"Bad embedding face {fr.id}: {e}")
                skipped += 1

        # For each image: filter by MIN_FACE_RATIO, then keep top MAX_FACES_PER_IMAGE
        for image_id, face_list in faces_by_image.items():
            # Sort by area ratio descending (largest face first)
            face_list.sort(key=lambda x: x[0], reverse=True)

            # Apply area filter — but always keep at least 1 face per image
            # (portrait shots where the face fills the frame pass trivially)
            passed = [item for item in face_list if item[0] >= MIN_FACE_RATIO]
            if not passed:
                # All faces were small — keep the single largest one anyway
                # (could be a small portrait photo shot from far away)
                passed = face_list[:1]
                logger.debug(f"🔍 Image {image_id}: all faces small, keeping largest "
                             f"(ratio={face_list[0][0]:.4f})")

            # Cap at MAX_FACES_PER_IMAGE
            kept = passed[:MAX_FACES_PER_IMAGE]
            dropped = len(face_list) - len(kept)
            if dropped > 0:
                bg_filtered += dropped
                logger.debug(f"🔍 Image {image_id}: kept {len(kept)}/{len(face_list)} faces "
                             f"(dropped {dropped} background faces, "
                             f"min_ratio={kept[-1][0]:.4f})")

            for _, fr, emb in kept:
                embeddings.append(emb)
                valid_faces.append(fr)

        logger.info(f"👥 {len(embeddings)} prominent face embeddings "
                    f"({skipped} invalid, {bg_filtered} background faces filtered out)")
        people_count = 0
        if embeddings:
            labels     = face_engine.cluster_faces(embeddings)
            person_map = {}
            for i, label in enumerate(labels):
                if label == -1: continue
                if label not in person_map:
                    p = Person(name=f"Person {label + 1}")
                    db.add(p); db.flush()
                    person_map[label] = p.id; people_count += 1
                valid_faces[i].person_id = person_map[label]
            db.commit()
            logger.info(f"✅ {people_count} people created")

            # ── Restore manually assigned names to matching new clusters ────
            if saved_names:
                restored = 0
                # Build map: face_id -> new person_id
                face_to_person = {}
                for face in db.query(DBFace).filter(DBFace.person_id != None).all():
                    face_to_person[face.id] = face.person_id

                for old_face_ids, saved_name in saved_names.items():
                    # Find which new person cluster has the most overlap with old faces
                    overlap_count = {}
                    for fid in old_face_ids:
                        pid = face_to_person.get(fid)
                        if pid:
                            overlap_count[pid] = overlap_count.get(pid, 0) + 1

                    if not overlap_count:
                        continue
                    # Best matching cluster
                    best_pid = max(overlap_count, key=overlap_count.get)
                    best_overlap = overlap_count[best_pid]
                    # Only restore if at least 30% of faces match
                    min_match = max(1, len(old_face_ids) * 0.30)
                    if best_overlap >= min_match:
                        person = db.query(Person).filter(Person.id == best_pid).first()
                        if person and re.match(r"^person\s+\d+$", (person.name or "").lower()):
                            person.name = saved_name
                            restored += 1
                            logger.info(f"✅ Restored name '{saved_name}' → Person {best_pid} ({best_overlap}/{len(old_face_ids)} faces matched)")
                db.commit()
                logger.info(f"🏷️  Restored {restored}/{len(saved_names)} manually assigned names")

            # Auto-name remaining unnamed clusters from BLIP captions / VQA
            _auto_name_people(db, person_map)
            db.commit()
        all_images   = db.query(DBImage).all()
        albums_count = 0
        if all_images:
            metadata = [{"id": img.id, "lat": img.lat or 0.0, "lon": img.lon or 0.0, "timestamp": img.timestamp} for img in all_images if img.timestamp]
            if metadata:
                album_labels = clustering_engine.detect_events(metadata)
                album_labels = clustering_engine.detect_events(metadata)
                album_map    = {}
                for i, label in enumerate(album_labels):
                    if label == -1:
                        continue
                    if label not in album_map:
                        cluster_meta = [metadata[j] for j, l in enumerate(album_labels) if l == label]
                        ts_list      = [m["timestamp"] for m in cluster_meta if m["timestamp"]]
                        start_d      = min(ts_list) if ts_list else None
                        end_d        = max(ts_list) if ts_list else None
                        if start_d:
                            if end_d and end_d.date() != start_d.date():
                                if end_d.month == start_d.month:
                                    title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%d, %Y')}"
                                else:
                                    title = f"{start_d.strftime('%b %d')} – {end_d.strftime('%b %d, %Y')}"
                            else:
                                # Single-day → canonical "Mar 26, 2026" title so all
                                # photos uploaded the same day share one album
                                title = start_d.strftime("%b %d, %Y")
                        else:
                            title = f"Event {label + 1}"
                        # Reuse existing album with same title to avoid duplicates
                        existing = (
                            db.query(Album)
                            .filter(Album.title == title, Album.type == "event")
                            .first()
                        )
                        if existing:
                            album_map[label] = existing.id
                        else:
                            new_album = Album(title=title, type="event", start_date=start_d, end_date=end_d)
                            db.add(new_album)
                            db.flush()
                            album_map[label] = new_album.id
                            albums_count += 1
                    db.query(DBImage).filter(DBImage.id == metadata[i]["id"]).update({"album_id": album_map[label]})
                db.commit()
                logger.info(f"✅ {albums_count} albums created")
        return {"status": "done", "people": people_count, "albums": albums_count}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Recluster failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG supported.")
    filename  = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(IMAGE_DIR, filename)
    db        = SessionLocal()
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        from PIL import Image as PILImage
        width = height = None
        avg_r = avg_g = avg_b = 0.0
        try:
            img_pil       = PILImage.open(file_path).convert("RGB")
            width, height = img_pil.size
            arr = np.array(img_pil)
            # Use dominant color (most common cluster) instead of plain average
            # Resize to 64x64 for speed, then find the peak of the color histogram
            import colorsys as _cs
            small = np.array(img_pil.resize((64, 64)), dtype=np.float32)
            r_vals = small[:,:,0].flatten()
            g_vals = small[:,:,1].flatten()
            b_vals = small[:,:,2].flatten()
            # Weight pixels by their saturation (ignore grey/white/black background)
            weights = []
            for ri, gi, bi in zip(r_vals, g_vals, b_vals):
                _, s, v = _cs.rgb_to_hsv(ri/255, gi/255, bi/255)
                weights.append(s * v + 0.01)  # saturated bright pixels count more
            weights = np.array(weights)
            total_w = weights.sum()
            avg_r = float(np.dot(r_vals, weights) / total_w)
            avg_g = float(np.dot(g_vals, weights) / total_w)
            avg_b = float(np.dot(b_vals, weights) / total_w)
        except Exception:
            pass
        clip_emb = None
        try:
            clip_emb = search_engine.get_image_embedding(file_path)
        except Exception:
            pass
        scene_label  = ""
        person_count = 0
        try:
            objects      = detector_engine.detect_objects(file_path, threshold=0.5)
            scene_label  = ", ".join(objects) if objects else ""
            person_count = detector_engine.detect_persons(file_path)
        except Exception:
            pass
        img_record = DBImage(filename=filename, original_path=file_path, timestamp=datetime_module.datetime.now(), width=width, height=height, avg_r=avg_r, avg_g=avg_g, avg_b=avg_b, scene_label=scene_label, person_count=person_count, ocr_text_enhanced="", ocr_keywords="[]", ocr_confidence=0.0, detected_language="en", caption_short="", caption_detailed="", caption_vqa="{}", quality_score=0.0, quality_level="Processing...", sharpness=0.0, exposure=0.0, contrast=0.0, composition=0.0, emotion_data="[]", dominant_emotion="neutral", face_emotion_count=0, aesthetic_score=0.0, aesthetic_rating="Processing...")
        db.add(img_record); db.flush()
        if clip_emb is not None:
            try:
                if search_engine.index is None:
                    search_engine.index = faiss.IndexIDMap(faiss.IndexFlatIP(clip_emb.shape[0]))
                new_vec = clip_emb.reshape(1, -1).astype("float32")
                faiss.normalize_L2(new_vec)
                search_engine.index.add_with_ids(new_vec, np.array([img_record.id], dtype="int64"))
                faiss.write_index(search_engine.index, FAISS_INDEX_PATH)
            except Exception as e:
                logger.warning(f"FAISS update failed: {e}")
        face_count = 0
        try:
            faces = face_engine.detect_faces(file_path)
            if faces and width and height:
                img_area = width * height
                # Sort faces by area descending (largest = most prominent first)
                def _face_area(f):
                    b = f["bbox"]  # [x1, y1, x2, y2]
                    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
                faces_sorted = sorted(faces, key=_face_area, reverse=True)
                # Keep only prominent faces (>= 0.5% of image area), max 4
                prominent = [f for f in faces_sorted
                             if _face_area(f) / img_area >= 0.005]
                if not prominent:
                    prominent = faces_sorted[:1]  # always keep largest face
                faces = prominent[:4]
            for face in faces:
                emb = face["embedding"].astype(np.float32)
                db.add(DBFace(image_id=img_record.id, bbox=_json.dumps(face["bbox"]), face_embedding=emb.tobytes()))
                face_count += 1
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
        db.commit()
        # ── Assign to today's event album immediately ──────────────────────────
        # Recluster runs in the background and takes a few seconds to complete.
        # Without this block a newly uploaded image has no album_id until
        # recluster finishes, so it would be invisible in the Albums tab.
        # We create (or reuse) a "Mar 26, 2026"-style event album right now
        # so the photo appears in Albums instantly. Recluster will merge it
        # into the correct cluster-based album when it completes.
        try:
            today       = datetime_module.datetime.now().date()
            today_title = today.strftime("%b %d, %Y")
            today_album = (
                db.query(Album)
                .filter(Album.title == today_title, Album.type == "event")
                .first()
            )
            if not today_album:
                today_album = Album(
                    title      = today_title,
                    type       = "event",
                    start_date = datetime_module.datetime.combine(today, datetime_module.time.min),
                    end_date   = datetime_module.datetime.combine(today, datetime_module.time.max),
                )
                db.add(today_album)
                db.flush()
            img_record.album_id = today_album.id
            db.commit()
            logger.info(f"📅 Image {img_record.id} → album '{today_title}' (id={today_album.id})")
        except Exception as _ae:
            logger.warning(f"Album auto-assign failed (non-fatal): {_ae}")
            try:
                db.rollback()
            except Exception:
                pass

        image_id = img_record.id
        if background_tasks is not None:
            background_tasks.add_task(_enrich_image, image_id, file_path)
            should_trigger_recluster(background_tasks)
        else:
            _enrich_image(image_id, file_path)
        logger.info(f"✅ Upload done: {filename} (id={image_id})")
        return {"status": "success", "id": image_id, "filename": filename, "person_count": person_count, "face_count": face_count, "quality_score": 0.0, "quality_level": "Processing...", "caption": "", "dominant_emotion": "neutral", "emotion_count": 0, "aesthetic_score": 0.0, "note": "ML enrichment running in background"}
    except Exception as e:
        db.rollback()
        if os.path.exists(file_path): os.remove(file_path)
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# REPROCESS
# ────────────────────────────────────────────────────────────────────────────
@app.post("/reprocess_images")
async def reprocess_images():
    db = SessionLocal()
    processed, failed = 0, 0
    try:
        images = db.query(DBImage).filter((DBImage.caption_short == None) | (DBImage.caption_short == "") | (DBImage.quality_level == "Processing...")).all()
        if not images:
            return {"success": True, "processed": 0, "failed": 0}
        for img in images:
            try:
                if not img.original_path or not os.path.exists(img.original_path):
                    failed += 1; continue
                _enrich_image(img.id, img.original_path); processed += 1
            except Exception:
                failed += 1
        return {"success": True, "processed": processed, "failed": failed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ────────────────────────────────────────────────────────────────────────────
# RECAPTION — re-run BLIP on images with missing or poor captions
# ────────────────────────────────────────────────────────────────────────────
@app.post("/recaption")
async def recaption_images(background_tasks: BackgroundTasks, force_all: bool = False):
    """
    Re-run BLIP captioning on images that:
    - Have no caption (caption_short is NULL or empty)
    - Have a placeholder/error caption
    - force_all=True: re-caption every image (useful after switching BLIP model)

    Runs in the background so the API returns immediately.
    Progress can be checked via GET /recaption/status
    """
    db = SessionLocal()
    try:
        if force_all:
            images = _live(db).all()
        else:
            # Only images missing captions
            images = _live(db).filter(
                (DBImage.caption_short == None) |
                (DBImage.caption_short == "") |
                (DBImage.caption_short == "Processing...")
            ).all()

        pairs = [
            (img.id, img.original_path or os.path.join(IMAGE_DIR, img.filename))
            for img in images
        ]
        # Filter to files that actually exist
        pairs = [(iid, fp) for iid, fp in pairs if os.path.exists(fp)]
    finally:
        db.close()

    if not pairs:
        return {"status": "nothing_to_do", "count": 0,
                "message": "All images already have captions. Use force_all=true to re-caption everything."}

    def _run_recaption(image_pairs):
        done = 0
        for img_id, fpath in image_pairs:
            try:
                db2 = SessionLocal()
                try:
                    caption_short    = captioning_engine.generate_caption(fpath, max_length=20)
                    caption_detailed = captioning_engine.generate_caption(fpath, max_length=60)
                    vqa_subject = captioning_engine.answer_visual_question(
                        fpath, "What is the main subject in this image?"
                    )
                    # Check if this image has people for VQA person question
                    img_chk = db2.query(DBImage).filter(DBImage.id == img_id).first()
                    vqa_person = ""
                    if img_chk and (img_chk.person_count or 0) > 0:
                        vqa_person = captioning_engine.answer_visual_question(fpath, "who is this?")
                        if vqa_person and vqa_person.lower().strip() in {
                            "man","woman","person","a man","a woman","a person","boy","girl",
                            "child","human","unknown","celebrity","actor","actress","no","yes",
                            "none","i don't know",
                        }:
                            vqa_person = ""

                    db2.query(DBImage).filter(DBImage.id == img_id).update({
                        "caption_short":    caption_short,
                        "caption_detailed": caption_detailed,
                        "caption_vqa":      _json.dumps({
                            "subject": vqa_subject,
                            "person":  vqa_person,
                        } if (vqa_subject or vqa_person) else {}),
                        "caption_timestamp": datetime_module.datetime.now(),
                    }, synchronize_session=False)
                    db2.commit()
                    done += 1
                    logger.info(f"✅ Recaptioned [{img_id}]: {caption_short}")
                finally:
                    db2.close()
            except Exception as e:
                logger.warning(f"⚠️ Recaption failed [{img_id}]: {e}")
        logger.info(f"✅ Recaption complete: {done}/{len(image_pairs)} images")

    background_tasks.add_task(_run_recaption, pairs)
    return {
        "status": "started",
        "count": len(pairs),
        "message": f"Re-captioning {len(pairs)} images in background. "
                   f"{'(all images)' if force_all else '(missing captions only)'}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)