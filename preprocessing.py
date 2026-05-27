import os
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

# =========================
# PATH SETTINGS
# =========================

DATASET_DIR = "clean_dataset"
OUTPUT_DIR = "processed_dataset_clean"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Image size for DenseNet121
IMG_SIZE = (224, 224)

# Split ratio
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Random seed for reproducibility
RANDOM_STATE = 42

# =========================
# READ CLASS NAMES
# =========================

class_names = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

print("\nDetected Classes:")
for idx, cls in enumerate(class_names):
    print(f"{idx}: {cls}")

# =========================
# COUNT ORIGINAL DATASET
# =========================

print("\nOriginal Dataset Distribution:")
original_counts = {}

for cls in class_names:
    class_path = os.path.join(DATASET_DIR, cls)
    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]
    original_counts[cls] = len(image_files)
    print(f"{cls}: {len(image_files)} images")

# =========================
# REMOVE OLD OUTPUT FOLDER
# =========================

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for cls in class_names:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

# =========================
# LOAD IMAGE PATHS AND LABELS
# =========================

all_image_paths = []
all_labels = []

for cls in class_names:
    class_path = os.path.join(DATASET_DIR, cls)
    
    for file in os.listdir(class_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            full_path = os.path.join(class_path, file)
            all_image_paths.append(full_path)
            all_labels.append(cls)

print(f"\nTotal Images Found: {len(all_image_paths)}")

# =========================
# TRAIN / VAL / TEST SPLIT
# =========================

# First split: Train 80%, Temp 20%
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_image_paths,
    all_labels,
    test_size=(1 - TRAIN_RATIO),
    stratify=all_labels,
    random_state=RANDOM_STATE
)

# Second split: Temp 20% into Validation 10% and Test 10%
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=RANDOM_STATE
)

print("\nDataset Split Completed Successfully!")
print(f"Training Images   : {len(train_paths)}")
print(f"Validation Images : {len(val_paths)}")
print(f"Testing Images    : {len(test_paths)}")

# =========================
# FUNCTION TO RESIZE AND SAVE IMAGES
# =========================

def process_and_save(image_paths, labels, destination_root):
    skipped = 0
    
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc=f"Processing {destination_root}"):
        try:
            img = cv2.imread(img_path)
            
            if img is None:
                skipped += 1
                continue
            
            img = cv2.resize(img, IMG_SIZE)
            
            filename = os.path.basename(img_path)
            save_path = os.path.join(destination_root, label, filename)
            
            cv2.imwrite(save_path, img)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            skipped += 1
    
    return skipped

# =========================
# PROCESS AND SAVE DATASET
# =========================

print("\nResizing and Saving Training Images...")
train_skipped = process_and_save(train_paths, train_labels, TRAIN_DIR)

print("\nResizing and Saving Validation Images...")
val_skipped = process_and_save(val_paths, val_labels, VAL_DIR)

print("\nResizing and Saving Testing Images...")
test_skipped = process_and_save(test_paths, test_labels, TEST_DIR)

print("\nImage Processing Completed!")
print(f"Skipped Training Images   : {train_skipped}")
print(f"Skipped Validation Images : {val_skipped}")
print(f"Skipped Testing Images    : {test_skipped}")

# =========================
# PRINT CLASS DISTRIBUTION
# =========================

def print_distribution(labels, title):
    print(f"\n{title}")
    counts = Counter(labels)
    for cls in class_names:
        print(f"{cls}: {counts[cls]}")

print_distribution(train_labels, "Training Set Distribution:")
print_distribution(val_labels, "Validation Set Distribution:")
print_distribution(test_labels, "Testing Set Distribution:")

# =========================
# PRINT FINAL CLASS LABEL MAPPING
# =========================

print("\nClass Label Mapping:")
for idx, cls in enumerate(class_names):
    print(f"{cls} --> {idx}")

# =========================
# DISPLAY SAMPLE IMAGES
# =========================

def show_sample_images(dataset_dir, class_names, samples_per_class=2):
    plt.figure(figsize=(12, 10))
    plot_index = 1
    
    for cls in class_names:
        class_path = os.path.join(dataset_dir, cls)
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        
        sample_images = random.sample(images, min(samples_per_class, len(images)))
        
        for img_file in sample_images:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(len(class_names), samples_per_class, plot_index)
            plt.imshow(img)
            plt.title(cls, fontsize=9)
            plt.axis("off")
            plot_index += 1
    
    plt.tight_layout()
    plt.show()

# Show sample images from training set
show_sample_images(TRAIN_DIR, class_names, samples_per_class=2)

# =========================
# FINAL SUCCESS MESSAGE
# =========================

print("\n========================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("Processed dataset saved in:", OUTPUT_DIR)
print("Ready for Kaggle training.")
print("========================================")