# Offline AI Smart Photo Management System (SmartGallery)

A fully offline alternatives to Google Photos / Apple Photos.

## Features
- **Semantic Search**: CLIP-powered search for concepts like "sunset", "dog", "party".
   - now augmented with **object tags** extracted using a Faster-RCNN detector. Queries such as "car" or "bicycle" will only return photos where that object was detected, dramatically reducing irrelevant results.
   - includes **color-aware queries** (e.g. "blue sky", "red car") based on average image color.
- **Face Recognition**: Automatically group photos by person.
- **OCR Search**: Find text inside tickets, documents, or signs.
- **Trip Detection**: Auto-detects events and clusters them into albums.
- **Duplicate Detection**: Find and remove identical or near-identical photos.
- **Map View**: Integrated world map for location-based exploration.
- **Voice Search**: Search using your voice (offline).

## Installation
1. Install Tesseract-OCR on your system.
2. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. (Optional) Download Vosk model for voice search:
   - Place in `models/vosk-model-small-en-us`

## Usage
1. Place images in `data/images`.
2. Run `python backend/build_index.py` to index (One-time).
3. Start API:
   ```bash
   cd backend
   python main.py
   ```
4. Start Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open `http://localhost:3000` in your browser.


## Tech Stack
- **Backend**: FastAPI, FAISS, SQLite, CLIP, Face_recognition, Tesseract.
- **Frontend**: React, Tailwind CSS, Lucide, Framer Motion, Leaflet.
