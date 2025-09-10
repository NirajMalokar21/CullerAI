import os
import shutil
import rawpy
from PIL import Image
import numpy as np
import cv2

RAW_EXTENSIONS = (".cr2", ".nef", ".arw")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp") + RAW_EXTENSIONS

# Feature extraction function
# Returns an array of features for the image provided
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sharpness
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness & noise
    brightness = np.mean(gray)
    noise = np.std(gray)

    # Color channel stats
    mean_b, mean_g, mean_r = np.mean(img, axis=(0,1))
    std_b, std_g, std_r = np.std(img, axis=(0,1))

    # Entropy
    hist, _ = np.histogram(gray, bins=256, range=(0,256), density=True)
    entropy_val = -np.sum(hist * np.log2(hist + 1e-7))

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return [
        sharpness, brightness, noise,
        mean_b, mean_g, mean_r,
        std_b, std_g, std_r,
        entropy_val, edge_density
    ]

# Load image (handles RAW and normal images)
def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTENSIONS:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess()
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return img
    else:
        img = cv2.imread(path)  # Reads in BGR format
        return img

# Gives the image a score after extracting features and calling the model
def score_image(model, path):
    img = load_image(path)
    features = extract_features(img)
    return model.predict([features])[0]

# Culls all the photos in the given source folder based on te threshold
def cull_photos(model, source_folder, dest_folder, threshold=0.6):
    os.makedirs(dest_folder, exist_ok=True)
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            full_path = os.path.join(source_folder, filename)
            score = score_image(model, full_path)
            if score >= threshold:
                shutil.copy(full_path, dest_folder)
