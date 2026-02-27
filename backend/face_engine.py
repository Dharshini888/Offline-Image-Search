
import numpy as np
import faiss
import os
import logging

logger = logging.getLogger("FaceEngine")

# InsightFace (ArcFace) — much more accurate than face_recognition/dlib
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not found. Run: pip install insightface onnxruntime")

FACE_INDEX_PATH = "../data/face_index.faiss"
FACE_DIM = 512  # ArcFace embedding dimension


class FaceEngine:
    def __init__(self):
        self.app = None
        self.face_index = None       # FAISS index for face embeddings
        self.face_id_map = []        # maps FAISS row → face DB id

        if INSIGHTFACE_AVAILABLE:
            self._load_model()
        self._load_face_index()

    def _load_model(self):
        """Load InsightFace ArcFace model."""
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",     # Best accuracy model (ArcFace ResNet100)
                providers=["CPUExecutionProvider"]  # Change to CUDAExecutionProvider for GPU
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace (ArcFace buffalo_l) loaded.")
        except Exception as e:
            logger.error(f"InsightFace model load failed: {e}")
            self.app = None

    def _load_face_index(self):
        """Load persisted FAISS face index from disk."""
        if os.path.exists(FACE_INDEX_PATH):
            try:
                self.face_index = faiss.read_index(FACE_INDEX_PATH)
                logger.info(f"Face FAISS index loaded ({self.face_index.ntotal} vectors).")
            except Exception as e:
                logger.error(f"Face index load failed: {e}")
                self.face_index = None
        else:
            # Initialize a new flat inner-product index
            self.face_index = faiss.IndexFlatIP(FACE_DIM)
            logger.info("New face FAISS index created.")

    def _save_face_index(self):
        """Persist FAISS face index to disk."""
        try:
            os.makedirs(os.path.dirname(FACE_INDEX_PATH), exist_ok=True)
            faiss.write_index(self.face_index, FACE_INDEX_PATH)
        except Exception as e:
            logger.error(f"Face index save failed: {e}")

    def detect_faces(self, image_path: str):
        """
        Detect faces in an image and return ArcFace 512-d embeddings.
        Returns: list of {"bbox": (x1,y1,x2,y2), "embedding": np.array(512,)}
        """
        if not INSIGHTFACE_AVAILABLE or self.app is None:
            return []
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return []
            faces = self.app.get(img)
            results = []
            for face in faces:
                emb = face.normed_embedding  # Already L2-normalized 512-d
                bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
                results.append({"bbox": bbox, "embedding": emb})
            return results
        except Exception as e:
            logger.error(f"Face detection failed for {image_path}: {e}")
            return []

    def add_to_index(self, embedding: np.ndarray, face_db_id: int):
        """Add a face embedding to the FAISS face index."""
        try:
            vec = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec)
            self.face_index.add(vec)
            self.face_id_map.append(face_db_id)
            self._save_face_index()
        except Exception as e:
            logger.error(f"Face index add failed: {e}")

    def rebuild_index(self, embeddings: list, face_db_ids: list):
        """
        Rebuild the FAISS face index from scratch.
        embeddings: list of np.array (512,)
        face_db_ids: corresponding Face.id values
        """
        if not embeddings:
            return
        self.face_index = faiss.IndexFlatIP(FACE_DIM)
        self.face_id_map = face_db_ids
        matrix = np.stack(embeddings).astype(np.float32)
        faiss.normalize_L2(matrix)
        self.face_index.add(matrix)
        self._save_face_index()
        logger.info(f"Face index rebuilt with {len(embeddings)} vectors.")

    def cluster_faces(self, embeddings: list):
        """
        DBSCAN clustering of face embeddings using cosine similarity.
        Tuned parameters for ArcFace 512-d embeddings:
        - eps=0.5: more lenient cosine distance threshold (original 0.4 was too strict)
        - min_samples=3: require at least 3 similar faces to form a cluster (prevents singleton clusters)
        Returns list of cluster labels (same length as embeddings).
        -1 = noise (unassigned face)
        """
        if not embeddings:
            return []
        from sklearn.cluster import DBSCAN
        matrix = np.stack(embeddings).astype(np.float32)
        # Normalize for cosine distance
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / (norms + 1e-10)
        # Increased eps from 0.4 to 0.5 (more lenient) and min_samples from 1 to 3
        # This prevents the same person from splitting into multiple clusters
        # model = DBSCAN(eps=0.5, min_samples=3, metric="cosine")
        model = DBSCAN(eps=0.25, min_samples=1, metric="cosine")
        labels = model.fit_predict(matrix)
        return labels


face_engine = FaceEngine()
