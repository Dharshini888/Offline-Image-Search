# Offline AI Smart Photo Management System (SmartGallery)

A fully offline, privacy-first alternative to Google Photos and Apple Photos. This project leverages state-of-the-art machine learning models to provide advanced photo search and management capabilities—entirely locally, without ever sending your data to the cloud.

## 📖 Project Overview

As our photo libraries grow, finding specific images becomes increasingly difficult. This project solves that problem by using deep learning to understand the contents of your photos. It acts as an intelligent, automated archivist. 

Whether you want to find all pictures of a "sunset on the beach," locate photos of a specific friend, read text off a photographed receipt, or automatically group your vacation photos into an album—Offline SmartGallery does it all securely on your own hardware.

### ✨ Key Features

- **Semantic Image Search**: Powered by OpenAI's CLIP model. Search for concepts like "birthday party," "dog playing," or "snowy mountains" using natural language.
- **Precision Object & Color Search**: Augmented with a Faster-RCNN object detector and color-awareness. A query for "red car" precisely returns photos containing red cars, reducing irrelevant results.
- **Advanced Face Recognition**: Uses **InsightFace (ArcFace)** and a dedicated FAISS index to accurately recognize, group, and tag people across thousands of photos, even grouping them into dedicated "People" profiles automatically.
- **OCR (Text) Search**: Uses Tesseract to extract text from images. Easily search for keywords inside tickets, documents, business cards, or signs.
- **Trip & Event Detection**: Automatically detects events chronologically and clusters them into distinct "Albums".
- **Duplicate Detection**: Finds and removes identical or near-identical photos to save storage space.
- **Map View**: Integrated interactive world map (Leaflet) for location-based exploration of geotagged images.
- **Voice Search**: Fully offline voice command processing using Vosk. Request searches just by speaking.

---

## 🛠 Tech Stack

- **Backend Base**: Python, FastAPI, SQLite
- **Vector Engine**: FAISS (for high-performance similarity search)
- **Machine Learning Models**:
  - **CLIP (OpenAI)**: Image & text embeddings for semantic search.
  - **Faster R-CNN**: Object detection.
  - **InsightFace (ArcFace)**: Face embeddings and recognition.
  - **DBSCAN**: Unsupervised clustering algorithm for face grouping and trip detection.
- **Utilities**: OpenCV, Pillow, PyTesseract (OCR)
- **Voice**: Vosk & PyAudio
- **Frontend**: React.js, Tailwind CSS, Lucide Icons, Framer Motion

---

## 📋 Prerequisites / What You Need

Before installing, ensure your system has the following installed:

1. **Python**: Version 3.9 through 3.11 recommended.
2. **Node.js & npm**: Required to run the React frontend.
3. **Tesseract-OCR**: Required for text-in-image searching.
   - **Windows**: Download the installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and ensure the installation path is added to your system's `PATH` environment variable.
4. **C++ Build Tools (Windows Users)**: Might be required for building `insightface` and `hnswlib`. Install **Visual Studio Build Tools** with the "Desktop development with C++" workload if the pip install fails.
5. **Microphone (Optional)**: If you plan to use the offline voice search feature.

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/offline-smart-gallery.git
cd offline-smart-gallery
```

### 2. Backend Setup
Install the Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### 3. Voice Model Setup (Optional)
To enable offline voice search:
1. Download a lightweight English model from [Vosk Models](https://alphacephei.com/vosk/models) (e.g., `vosk-model-small-en-us-0.15.zip`).
2. Extract the folder and place it in the project root under `models/vosk-model-small-en-us`.

### 4. Frontend Setup
Install the JavaScript dependencies:
```bash
cd ../frontend
npm install
```

---

## 💻 Usage

### 1. Adding Images
Place the photos you want to manage inside the data directory: `data/images`.

### 2. Building the Index (One-Time / When adding new photos)
Before you can search, the AI needs to analyze your images and build the vector databases:
```bash
cd backend
python build_index.py
```
*Note: This process may take a while depending on the number of photos and your computer's specs.*

### 3. Start the Backend API
```bash
cd backend
python main.py
```

### 4. Start the Frontend Application
Open a new terminal window:
```bash
cd frontend
npm run dev
```

Finally, open your browser and navigate to `http://localhost:3000` to start exploring your automated, offline photo gallery!
