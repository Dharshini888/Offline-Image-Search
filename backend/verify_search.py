import faiss
import sqlite3
import torch
import clip
import numpy as np
from PIL import Image
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-B/32"
INDEX_PATH = "../data/index.faiss"
DB_PATH = "../data/db.sqlite"

def test_search():
    if not os.path.exists(INDEX_PATH):
        print("Error: Index not found.")
        return

    print(f"Loading CLIP {CLIP_MODEL}...")
    model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
    
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    print(f"Index count: {index.ntotal}")

    query = "nature"
    print(f"Searching for: '{query}'")
    
    tokens = clip.tokenize([query]).to(DEVICE)
    with torch.no_grad():
        query_emb = model.encode_text(tokens)
        query_emb /= query_emb.norm(dim=-1, keepdim=True)
    
    query_emb = query_emb.cpu().numpy().astype('float32')
    
    # Search top 5
    distances, indices = index.search(query_emb, 5)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\nResults:")
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: continue
        
        cursor.execute("SELECT filename FROM images WHERE id = ?", (int(idx),))
        row = cursor.fetchone()
        filename = row[0] if row else "UNKNOWN"
        print(f"ID: {idx}, Dist: {dist:.4f}, File: {filename}")
        
    conn.close()

if __name__ == "__main__":
    test_search()
