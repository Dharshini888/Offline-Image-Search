# # #!/usr/bin/env python3
# # """Small helper to run a local search using the same logic as the API and print diagnostics."""
# # import os
# # import sys
# # import numpy as np
# # import faiss
# # from database import SessionLocal, Image as DBImage, init_db
# # from search_engine import search_engine, resolve_query

# # # Match thresholds from main.py (defaults updated for stricter matches)
# # CLIP_SCORE_MIN = float(os.environ.get("CLIP_SCORE_MIN", 0.55))
# # FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.40))

# # def local_search(query, top_k=10):
# #     init_db()
# #     db = SessionLocal()
# #     try:
# #         # Step 1: Load FAISS index if needed
# #         if search_engine.index is None:
# #             index_path = "../data/index.faiss"
# #             if os.path.exists(index_path):
# #                 try:
# #                     search_engine.index = faiss.read_index(index_path)
# #                     print(f"✅ Loaded FAISS index ({search_engine.index.ntotal} vectors)")
# #                 except Exception as e:
# #                     print(f"❌ Failed to load FAISS index: {e}")
# #                     return
# #             else:
# #                 print(f"❌ FAISS index not found at {index_path}")
# #                 return
        
# #         # Step 2: Process query
# #         processed = resolve_query(query)
# #         print(f"Query: '{query}' -> expanded: '{processed}'")

# #         # Step 3: Get text embedding
# #         try:
# #             emb = search_engine.get_text_embedding(processed, use_prompt_ensemble=True)
# #             if emb is None:
# #                 print("❌ Failed to get text embedding from CLIP")
# #                 return
# #             print(f"✅ Got text embedding (dims: {emb.shape[0]})")
# #         except Exception as e:
# #             print(f"❌ Error getting text embedding: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return

# #         q = emb.reshape(1, -1).astype('float32')
# #         faiss.normalize_L2(q)
# #         candidate_k = min(top_k * 8, 250)  # keep consistent with server logic
# #         dists, idxs = search_engine.index.search(q, candidate_k)
# #         valid_count = len([i for i in idxs[0] if i != -1])
# #         print(f"✅ FAISS search found {valid_count} candidates\n")

# #         results = []
# #         skipped_clip = 0
# #         skipped_final = 0
        
# #         for raw_sim, idx in zip(dists[0], idxs[0]):
# #             if idx == -1:
# #                 continue
# #             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
# #             if not img:
# #                 continue

# #             clip_score = max(0.0, min(1.0, (float(raw_sim) + 1.0) / 2.0))
            
# #             # CLIP filtering
# #             if clip_score < CLIP_SCORE_MIN:
# #                 skipped_clip += 1
# #                 continue
                
# #             ocr_text = (img.ocr_text or "").lower()
# #             ocr_bonus = 0.0
# #             words = processed.lower().split()
# #             sig = [w for w in words if len(w) > 2]
# #             if sig:
# #                 matches = sum(1 for w in sig if w in ocr_text)
# #                 ocr_bonus = min(matches / len(sig), 1.0)

# #             color_bonus = 0.0

# #             # tag bonus (if image has scene_label)
# #             tag_bonus = 0.0
# #             if img.scene_label:
# #                 tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
# #                 if any(w in tags for w in processed.lower().split()):
# #                     tag_bonus = 1.0

# #             final = search_engine.hybrid_rank(clip_score, ocr_bonus=ocr_bonus, color_bonus=color_bonus, tag_bonus=tag_bonus)
            
# #             # Final score filtering
# #             if final < FINAL_SCORE_MIN:
# #                 skipped_final += 1
# #                 continue
            
# #             results.append((final, raw_sim, clip_score, ocr_bonus, img.filename))

# #         results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
        
# #         if not results:
# #             print(f"⚠️  No results found (skipped {skipped_clip} for CLIP, {skipped_final} for final score)")
# #             return
        
# #         print(f"🔍 Top {len(results)} Results (filtered: {skipped_clip} low CLIP, {skipped_final} low final):")
# #         print("-" * 110)
# #         for rank, (final, raw_sim, clip_score, ocr_bonus, fname) in enumerate(results, start=1):
# #             print(f"{rank:2d}.  {final:.4f}  |  RAW={raw_sim:+.4f}  CLIP={clip_score:.3f}  OCR={ocr_bonus:.2f}  |  {fname}")

# #     finally:
# #         db.close()
# #         print("-" * 110)

# # if __name__ == '__main__':
# #     import sys
# #     import argparse
# #     p = argparse.ArgumentParser()
# #     p.add_argument('query', nargs='*', default=['cat'])  # Default to 'cat' if not provided
# #     p.add_argument('--k', type=int, default=10)
# #     args = p.parse_args()
# #     query = ' '.join(args.query) if args.query else 'cat'
# #     print(f"💡 Running search test locally...\n")
# #     local_search(query, top_k=args.k)
# #!/usr/bin/env python3
# """Small helper to run a local search using the same logic as the API and print diagnostics."""
# import os
# import sys
# import numpy as np
# import faiss
# from database import SessionLocal, Image as DBImage, init_db
# from search_engine import search_engine, resolve_query

# # FIXED: Lowered thresholds for better recall
# CLIP_SCORE_MIN = float(os.environ.get("CLIP_SCORE_MIN", 0.14))
# FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.14))

# def local_search(query, top_k=10):
#     init_db()
#     db = SessionLocal()
#     try:
#         # Step 1: Load FAISS index if needed
#         if search_engine.index is None:
#             index_path = "../data/index.faiss"
#             if os.path.exists(index_path):
#                 try:
#                     search_engine.index = faiss.read_index(index_path)
#                     print(f"✅ Loaded FAISS index ({search_engine.index.ntotal} vectors)")
#                 except Exception as e:
#                     print(f"❌ Failed to load FAISS index: {e}")
#                     return
#             else:
#                 print(f"❌ FAISS index not found at {index_path}")
#                 return
        
#         # Step 2: Process query
#         processed = resolve_query(query)
#         print(f"Query: '{query}' -> expanded: '{processed}'")

#         # Step 3: Get text embedding
#         try:
#             emb = search_engine.get_text_embedding(processed, use_prompt_ensemble=True)
#             if emb is None:
#                 print("❌ Failed to get text embedding from CLIP")
#                 return
#             print(f"✅ Got text embedding (dims: {emb.shape[0]})")
#         except Exception as e:
#             print(f"❌ Error getting text embedding: {e}")
#             import traceback
#             traceback.print_exc()
#             return

#         q = emb.reshape(1, -1).astype('float32')
#         faiss.normalize_L2(q)
#         candidate_k = min(top_k * 8, 250)  # keep consistent with server logic
#         dists, idxs = search_engine.index.search(q, candidate_k)
#         valid_count = len([i for i in idxs[0] if i != -1])
#         print(f"✅ FAISS search found {valid_count} candidates\n")

#         results = []
#         skipped_clip = 0
#         skipped_final = 0
        
#         for raw_sim, idx in zip(dists[0], idxs[0]):
#             if idx == -1:
#                 continue
#             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
#             if not img:
#                 continue

#             # FIXED: Use FAISS score directly (already normalized to [0, 1])
#             # OLD BUG: clip_score = max(0.0, min(1.0, (float(raw_sim) + 1.0) / 2.0))
#             # That formula assumed raw_sim was in [-1, 1], but FAISS returns [0, 1]!
#             clip_score = float(raw_sim)
            
#             # CLIP filtering (with lower threshold now)
#             if clip_score < CLIP_SCORE_MIN:
#                 skipped_clip += 1
#                 continue
                
#             # OCR bonus
#             ocr_text = (img.ocr_text or "").lower()
#             ocr_bonus = 0.0
#             words = processed.lower().split()
#             sig = [w for w in words if len(w) > 2]
#             if sig:
#                 matches = sum(1 for w in sig if w in ocr_text)
#                 ocr_bonus = min(matches / len(sig), 1.0)

#             # Color bonus (if image has average color)
#             color_bonus = 0.0
#             if img.avg_r is not None and img.avg_g is not None and img.avg_b is not None:
#                 # Simple heuristic: dark/bright images score differently
#                 brightness = (img.avg_r + img.avg_g + img.avg_b) / 3.0 / 255.0
#                 # Could be expanded to actual color matching for queries like "red sunset"
#                 color_bonus = 0.0  # Disabled for now

#             # Tag bonus (if image has scene_label/objects detected)
#             tag_bonus = 0.0
#             if img.scene_label:
#                 tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
#                 # Match any query word with detected tags
#                 if any(w in tags for w in processed.lower().split()):
#                     tag_bonus = 1.0

#             # Compute final score using hybrid ranking
#             final = search_engine.hybrid_rank(
#                 clip_score,
#                 ocr_bonus=ocr_bonus,
#                 color_bonus=color_bonus,
#                 tag_bonus=tag_bonus
#             )
            
#             # Final score filtering
#             if final < FINAL_SCORE_MIN:
#                 skipped_final += 1
#                 continue
            
#             results.append((final, raw_sim, clip_score, ocr_bonus, img.filename))

#         results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
        
#         if not results:
#             print(f"⚠️  No results found (skipped {skipped_clip} for CLIP, {skipped_final} for final score)")
#             print(f"\nDebug info:")
#             print(f"  - CLIP_SCORE_MIN: {CLIP_SCORE_MIN}")
#             print(f"  - FINAL_SCORE_MIN: {FINAL_SCORE_MIN}")
#             print(f"  - Try lowering thresholds or check that images were indexed")
#             return
        
#         print(f"🔍 Top {len(results)} Results (filtered: {skipped_clip} low CLIP, {skipped_final} low final):")
#         print("-" * 110)
#         print(f"{'Rank':<4} {'Final':<8} {'RAW':<10} {'CLIP':<8} {'OCR':<6} {'Filename':<60}")
#         print("-" * 110)
#         for rank, (final, raw_sim, clip_score, ocr_bonus, fname) in enumerate(results, start=1):
#             print(f"{rank:<4} {final:.4f}   {raw_sim:+.4f}      {clip_score:.3f}    {ocr_bonus:.2f}   {fname}")

#     finally:
#         db.close()
#         print("-" * 110)

# if __name__ == '__main__':
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument('query', nargs='*', default=['dog'])  # Default to 'dog'
#     p.add_argument('--k', type=int, default=10)
#     args = p.parse_args()
#     query = ' '.join(args.query) if args.query else 'dog'
#     print(f"💡 Running search test locally...\n")
#     local_search(query, top_k=args.k)


# #!/usr/bin/env python3
# """Small helper to run a local search using the same logic as the API and print diagnostics."""
# import os
# import sys
# import numpy as np
# import faiss
# from database import SessionLocal, Image as DBImage, init_db
# from search_engine import search_engine, resolve_query

# # Match thresholds from main.py (defaults updated for stricter matches)
# CLIP_SCORE_MIN = float(os.environ.get("CLIP_SCORE_MIN", 0.55))
# FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.40))

# def local_search(query, top_k=10):
#     init_db()
#     db = SessionLocal()
#     try:
#         # Step 1: Load FAISS index if needed
#         if search_engine.index is None:
#             index_path = "../data/index.faiss"
#             if os.path.exists(index_path):
#                 try:
#                     search_engine.index = faiss.read_index(index_path)
#                     print(f"✅ Loaded FAISS index ({search_engine.index.ntotal} vectors)")
#                 except Exception as e:
#                     print(f"❌ Failed to load FAISS index: {e}")
#                     return
#             else:
#                 print(f"❌ FAISS index not found at {index_path}")
#                 return
        
#         # Step 2: Process query
#         processed = resolve_query(query)
#         print(f"Query: '{query}' -> expanded: '{processed}'")

#         # Step 3: Get text embedding
#         try:
#             emb = search_engine.get_text_embedding(processed, use_prompt_ensemble=True)
#             if emb is None:
#                 print("❌ Failed to get text embedding from CLIP")
#                 return
#             print(f"✅ Got text embedding (dims: {emb.shape[0]})")
#         except Exception as e:
#             print(f"❌ Error getting text embedding: {e}")
#             import traceback
#             traceback.print_exc()
#             return

#         q = emb.reshape(1, -1).astype('float32')
#         faiss.normalize_L2(q)
#         candidate_k = min(top_k * 8, 250)  # keep consistent with server logic
#         dists, idxs = search_engine.index.search(q, candidate_k)
#         valid_count = len([i for i in idxs[0] if i != -1])
#         print(f"✅ FAISS search found {valid_count} candidates\n")

#         results = []
#         skipped_clip = 0
#         skipped_final = 0
        
#         for raw_sim, idx in zip(dists[0], idxs[0]):
#             if idx == -1:
#                 continue
#             img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
#             if not img:
#                 continue

#             clip_score = max(0.0, min(1.0, (float(raw_sim) + 1.0) / 2.0))
            
#             # CLIP filtering
#             if clip_score < CLIP_SCORE_MIN:
#                 skipped_clip += 1
#                 continue
                
#             ocr_text = (img.ocr_text or "").lower()
#             ocr_bonus = 0.0
#             words = processed.lower().split()
#             sig = [w for w in words if len(w) > 2]
#             if sig:
#                 matches = sum(1 for w in sig if w in ocr_text)
#                 ocr_bonus = min(matches / len(sig), 1.0)

#             color_bonus = 0.0

#             # tag bonus (if image has scene_label)
#             tag_bonus = 0.0
#             if img.scene_label:
#                 tags = [t.strip().lower() for t in img.scene_label.split(",") if t.strip()]
#                 if any(w in tags for w in processed.lower().split()):
#                     tag_bonus = 1.0

#             final = search_engine.hybrid_rank(clip_score, ocr_bonus=ocr_bonus, color_bonus=color_bonus, tag_bonus=tag_bonus)
            
#             # Final score filtering
#             if final < FINAL_SCORE_MIN:
#                 skipped_final += 1
#                 continue
            
#             results.append((final, raw_sim, clip_score, ocr_bonus, img.filename))

#         results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
        
#         if not results:
#             print(f"⚠️  No results found (skipped {skipped_clip} for CLIP, {skipped_final} for final score)")
#             return
        
#         print(f"🔍 Top {len(results)} Results (filtered: {skipped_clip} low CLIP, {skipped_final} low final):")
#         print("-" * 110)
#         for rank, (final, raw_sim, clip_score, ocr_bonus, fname) in enumerate(results, start=1):
#             print(f"{rank:2d}.  {final:.4f}  |  RAW={raw_sim:+.4f}  CLIP={clip_score:.3f}  OCR={ocr_bonus:.2f}  |  {fname}")

#     finally:
#         db.close()
#         print("-" * 110)

# if __name__ == '__main__':
#     import sys
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument('query', nargs='*', default=['cat'])  # Default to 'cat' if not provided
#     p.add_argument('--k', type=int, default=10)
#     args = p.parse_args()
#     query = ' '.join(args.query) if args.query else 'cat'
#     print(f"💡 Running search test locally...\n")
#     local_search(query, top_k=args.k)
#!/usr/bin/env python3
"""Small helper to run a local search using the same logic as the API and print diagnostics."""
import os
import sys
import numpy as np
import faiss
from database import SessionLocal, Image as DBImage, init_db
from search_engine import search_engine, resolve_query

# FIXED: Lowered thresholds for better recall
CLIP_SCORE_MIN = float(os.environ.get("CLIP_SCORE_MIN", 0.22))
FINAL_SCORE_MIN = float(os.environ.get("FINAL_SCORE_MIN", 0.22))

def local_search(query, top_k=10):
    init_db()
    db = SessionLocal()
    try:
        # Step 1: Load FAISS index if needed
        if search_engine.index is None:
            index_path = "../data/index.faiss"
            if os.path.exists(index_path):
                try:
                    search_engine.index = faiss.read_index(index_path)
                    print(f"✅ Loaded FAISS index ({search_engine.index.ntotal} vectors)")
                except Exception as e:
                    print(f"❌ Failed to load FAISS index: {e}")
                    return
            else:
                print(f"❌ FAISS index not found at {index_path}")
                return
        
        # Step 2: Process query
        processed = resolve_query(query)
        print(f"Query: '{query}' -> expanded: '{processed}'")

        # Step 3: Get text embedding
        try:
            emb = search_engine.get_text_embedding(processed, use_prompt_ensemble=True)
            if emb is None:
                print("❌ Failed to get text embedding from CLIP")
                return
            print(f"✅ Got text embedding (dims: {emb.shape[0]})")
        except Exception as e:
            print(f"❌ Error getting text embedding: {e}")
            import traceback
            traceback.print_exc()
            return

        q = emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)
        candidate_k = min(top_k * 8, 250)  # keep consistent with server logic
        dists, idxs = search_engine.index.search(q, candidate_k)
        valid_count = len([i for i in idxs[0] if i != -1])
        print(f"✅ FAISS search found {valid_count} candidates\n")

        results = []
        skipped_clip = 0
        skipped_final = 0
        
        for raw_sim, idx in zip(dists[0], idxs[0]):
            if idx == -1:
                continue
            img = db.query(DBImage).filter(DBImage.id == int(idx)).first()
            if not img:
                continue

            # FIXED: Use FAISS score directly (already normalized to [0, 1])
            # OLD BUG: clip_score = max(0.0, min(1.0, (float(raw_sim) + 1.0) / 2.0))
            # That formula assumed raw_sim was in [-1, 1], but FAISS returns [0, 1]!
            clip_score = float(raw_sim)
            
            # CLIP filtering (with lower threshold now)
            if clip_score < CLIP_SCORE_MIN:
                skipped_clip += 1
                continue
                
            # OCR bonus
            ocr_text = (img.ocr_text or "").lower()
            ocr_bonus = 0.0
            words = processed.lower().split()
            sig = [w for w in words if len(w) > 2]
            if sig:
                matches = sum(1 for w in sig if w in ocr_text)
                ocr_bonus = min(matches / len(sig), 1.0)

            # Color bonus (if image has average color)
            color_bonus = 0.0
            if img.avg_r is not None and img.avg_g is not None and img.avg_b is not None:
                # Simple heuristic: dark/bright images score differently
                brightness = (img.avg_r + img.avg_g + img.avg_b) / 3.0 / 255.0
                # Could be expanded to actual color matching for queries like "red sunset"
                color_bonus = 0.0  # Disabled for now

            # Tag bonus — exact word match only
            # Old: `w in tags` (substring) matched "man" inside "woman", "dog" in "hotdog"
            # Fix: flatten tags into individual words and do exact set membership
            tag_bonus = 0.0
            if img.scene_label:
                tag_words = set()
                for tag in img.scene_label.split(","):
                    for word in tag.strip().lower().split():
                        tag_words.add(word)
                query_sig = [w for w in processed.lower().split() if len(w) > 2]
                if any(w in tag_words for w in query_sig):
                    tag_bonus = 1.0

            # Compute final score using hybrid ranking
            final = search_engine.hybrid_rank(
                clip_score,
                ocr_bonus=ocr_bonus,
                color_bonus=color_bonus,
                tag_bonus=tag_bonus
            )
            
            # Final score filtering
            if final < FINAL_SCORE_MIN:
                skipped_final += 1
                continue
            
            results.append((final, raw_sim, clip_score, ocr_bonus, img.filename))

        results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
        
        if not results:
            print(f"⚠️  No results found (skipped {skipped_clip} for CLIP, {skipped_final} for final score)")
            print(f"\nDebug info:")
            print(f"  - CLIP_SCORE_MIN: {CLIP_SCORE_MIN}")
            print(f"  - FINAL_SCORE_MIN: {FINAL_SCORE_MIN}")
            print(f"  - Try lowering thresholds or check that images were indexed")
            return
        
        print(f"🔍 Top {len(results)} Results (filtered: {skipped_clip} low CLIP, {skipped_final} low final):")
        print("-" * 110)
        print(f"{'Rank':<4} {'Final':<8} {'RAW':<10} {'CLIP':<8} {'OCR':<6} {'Filename':<60}")
        print("-" * 110)
        for rank, (final, raw_sim, clip_score, ocr_bonus, fname) in enumerate(results, start=1):
            print(f"{rank:<4} {final:.4f}   {raw_sim:+.4f}      {clip_score:.3f}    {ocr_bonus:.2f}   {fname}")

    finally:
        db.close()
        print("-" * 110)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('query', nargs='*', default=['dog'])  # Default to 'dog'
    p.add_argument('--k', type=int, default=10)
    args = p.parse_args()
    query = ' '.join(args.query) if args.query else 'dog'
    print(f"💡 Running search test locally...\n")
    local_search(query, top_k=args.k)