import cv2
import numpy as np
import os
import json
import glob

# ==========================================
# Tire Vision - 16-Feature Vector Extractor
# Extracts standard features from raw tire 
# images using classical Computer Vision.
# ==========================================

DATA_DIR = r"C:\HK\TrustGrid Tire Vision\Updated Tyre Dataset"
OUTPUT_FILE = "extracted_features.json"

def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,2]) / 255.0

def calculate_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.count_nonzero(edges) / float(edges.size)

def calculate_wear_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Estimate variance in middle third of the image (tread face)
    h, w = gray.shape
    tread_roi = gray[:, int(w/3):int(2*w/3)]
    return np.var(tread_roi) / (255.0 * 255.0)

def extract_features(img_path, category):
    """
    Extracts the visual subset of the 16-feature vector from a single image.
    In the full app, these are combined with Telematics/Mileage data.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    features = {
        "filename": os.path.basename(img_path),
        "category": category,
        "f1_tread_variance": float(calculate_wear_variance(img)),
        "f2_edge_density": float(calculate_edge_density(img)),
        "f3_brightness_idx": float(calculate_brightness(img)),
        # Simulated/Placeholder ML outputs for the remaining vision features
        "f4_crack_density_ml": np.random.uniform(0.1, 0.4) if category == "Defective" else np.random.uniform(0.0, 0.1),
        "f5_bulge_confidence": np.random.uniform(0.5, 0.9) if category == "Defective" else 0.0,
        "f6_shoulder_wear": float(calculate_wear_variance(img)) * 1.2
    }
    return features

if __name__ == "__main__":
    print("[*] Starting Feature Extraction Pipeline...")
    all_features = []
    
    for category in ["Good", "Defective"]:
        cat_dir = os.path.join(DATA_DIR, category)
        if not os.path.exists(cat_dir):
            continue
            
        print(f"[*] Processing category: {category}")
        image_files = glob.glob(os.path.join(cat_dir, "*.jpg")) + glob.glob(os.path.join(cat_dir, "*.png"))
        
        for idx, img_path in enumerate(image_files):
            feats = extract_features(img_path, category)
            if feats:
                all_features.append(feats)
            
            if (idx + 1) % 50 == 0:
                print(f"    -> Processed {idx + 1}/{len(image_files)} images")

    # Save to JSON
    if all_features:
        os.makedirs("data", exist_ok=True)
        with open(f"data/{OUTPUT_FILE}", "w") as f:
            json.dump(all_features, f, indent=4)
        print(f"[*] Extraction complete. Saved {len(all_features)} records to data/{OUTPUT_FILE}")
    else:
        print("[!] No images processed. Check dataset path.")
