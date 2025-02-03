import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import models, transforms
from PIL import Image

# Set device for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
model = model.to(device).eval()  # Set to evaluation mode

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Feature extraction function
def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
        with torch.no_grad():  # Disable gradient calculation for inference
            feature = model(image).cpu().numpy().flatten()  # Flatten the feature vector
        return feature
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Build image feature matrix
def build_image_features(image_dir, metadata_file):
    metadata = pd.read_csv(metadata_file)
    image_paths = [os.path.join(image_dir, image_name) for image_name in metadata['image_name']]

    features = []
    for image_path in tqdm(image_paths, desc="Extracting Features"):
        if os.path.exists(image_path):
            feature = extract_features(image_path)
            if feature is not None:
                features.append(feature)
        else:
            print(f"Image {image_path} not found. Skipping.")

    features = np.array(features)
    return image_paths, features

# Save features and image paths
image_dir = 'images'  # Path to your images directory
metadata_file = 'cleaned_results.csv'  # Path to the cleaned metadata CSV
image_paths, features = build_image_features(image_dir, metadata_file)

np.save('features.npy', features)
np.save('image_paths.npy', image_paths)

print("Feature extraction complete. 'features.npy' and 'image_paths.npy' saved.")
