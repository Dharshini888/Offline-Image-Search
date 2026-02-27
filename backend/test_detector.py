from detector_engine import detector_engine
import os

# Find a sample image
IMAGE_DIR = "../data/images"
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("No images found to test.")
else:
    sample_path = os.path.join(IMAGE_DIR, image_files[0])
    print(f"Testing person detection on: {sample_path}")
    count = detector_engine.detect_persons(sample_path)
    print(f"Detected persons: {count}")
    # also run object detection and show categories
    labels = detector_engine.detect_objects(sample_path)
    print(f"Detected objects/tags: {labels}")
