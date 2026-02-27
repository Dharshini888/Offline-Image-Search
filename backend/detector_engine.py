import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import torchvision.transforms as T
import logging

logger = logging.getLogger("DetectorEngine")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DetectorEngine:
    def __init__(self):
        logger.info(f"Loading Faster R-CNN on {DEVICE}...")
        # Load pre-trained model and capture category names
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.to(DEVICE)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])

        # list of COCO category names (index corresponds to label id)
        # first element is '__background__'
        self.categories = self.weights.meta.get('categories', [])

    def detect_persons(self, image_path):
        """Returns the number of persons detected in the image."""
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).to(DEVICE)
            
            with torch.no_grad():
                predictions = self.model([img_tensor])
            
            # Label 1 is 'person' in COCO dataset
            labels = predictions[0]['labels']
            scores = predictions[0]['scores']
            
            # Filter for person label and high confidence score (> 0.5)
            person_count = ((labels == 1) & (scores > 0.5)).sum().item()
            
            return int(person_count)
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return 0

    def detect_objects(self, image_path, threshold: float = 0.5):
        """Run object detection and return a list of category names above threshold."""
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).to(DEVICE)
            with torch.no_grad():
                predictions = self.model([img_tensor])

            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            objects = []
            for lbl, score in zip(labels, scores):
                if score >= threshold and lbl < len(self.categories):
                    name = self.categories[int(lbl)]
                    if name != '__background__':
                        objects.append(name)
            # deduplicate
            return list(set(objects))
        except Exception as e:
            logger.error(f"Object detection failed for {image_path}: {e}")
            return []

    def detect_persons(self, image_path):
        """Returns the number of persons detected in the image."""
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).to(DEVICE)
            
            with torch.no_grad():
                predictions = self.model([img_tensor])
            
            # Label 1 is 'person' in COCO dataset
            labels = predictions[0]['labels']
            scores = predictions[0]['scores']
            
            # Filter for person label and high confidence score (> 0.5)
            person_count = ((labels == 1) & (scores > 0.5)).sum().item()
            
            return int(person_count)
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return 0

detector_engine = DetectorEngine()
