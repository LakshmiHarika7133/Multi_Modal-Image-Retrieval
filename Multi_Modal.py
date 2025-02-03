import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Set Streamlit page config
st.set_page_config(page_title="Multi Modal Image Retrieval", layout="wide")

# Load pre-saved features and paths
@st.cache_data
def load_image_features():
    image_features = np.load("features.npy")
    image_paths = np.load("image_paths.npy", allow_pickle=True)
    return image_paths, image_features

image_paths, features = load_image_features()

# Load text captions
@st.cache_data
def load_text_captions():
    df = pd.read_csv("cleaned_results.csv")  # Ensure the cleaned CSV file exists
    return df

df = load_text_captions()

# Ensure the 'uploads' directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Home", "Text-to-Image", "Image-to-Image"])

# Shared function to display images with similarity
def display_results(results, max_images=10):
    cols = st.columns(5)  # Display up to 5 images per row
    for i, (image_path, similarity) in enumerate(results[:max_images]):  # Limit to max_images
        if similarity > 0.1:  # Only show results with similarity > 0.5
            with cols[i % 5]:  # Wrap to next row every 5 images
                img = Image.open(image_path)
                img = img.resize((150, 150))  # Resize image for smaller display
                st.image(img, caption=f"Similarity: {similarity:.2f}", use_column_width=True)

# Home Page
if page == "Home":
    st.title("Welcome to the Multi Modal Image Retrieval System")
    st.markdown(
        """
        This is a **Multi Modal Image Retrieval** application that supports two main functionalities:
        - **Text-to-Image Retrieval**: Retrieve images based on a text query.
        - **Image-to-Image Retrieval**: Find visually similar images by uploading an example image.
        
        **How to Use**:
        - Navigate to the desired page using the sidebar.
        - Follow the instructions on the selected page.
        """
    )
    st.image("background.jpg", use_column_width=True)  # Replace with your background image

# Text-to-Image Page
elif page == "Text-to-Image":
    st.title("Text-to-Image Retrieval")
    st.markdown(
        """
        **Instructions**:
        - Enter a descriptive text query in the input box below.
        - The system will retrieve up to **10 images** that best match your query based on their captions.
        - Only images with a similarity score greater than **0.1** will be displayed.
        """
    )
    query = st.text_input("Enter a text query:")
    
    if query:
        # Vectorize query and compute similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(df["caption"])
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        df["similarity"] = similarities
        results = df[df["similarity"] > 0.1][["image_name", "similarity"]]
        results = results.sort_values("similarity", ascending=False)

        if results.empty:
            st.warning("No results found with similarity > 0.1.")
        else:
            st.write(f"Top {min(len(results), 10)} results:")
            results_to_display = [(os.path.join("images", row["image_name"]), row["similarity"]) for _, row in results.iterrows()]
            display_results(results_to_display, max_images=10)

# Image-to-Image Page
elif page == "Image-to-Image":
    st.title("Image-to-Image Retrieval")
    st.markdown(
        """
        **Instructions**:
        - Upload an image from your device.
        - The system will find up to **10 visually similar images** from the dataset.
        - Only images with a similarity score greater than **0.1** will be displayed.
        """
    )
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        query_image_path = os.path.join("uploads", uploaded_file.name)
        # Ensure the uploaded image is saved properly
        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Feature extraction for uploaded image
        from torchvision import models, transforms
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def extract_feature(image_path):
            img = Image.open(image_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model(img).cpu().numpy().flatten()
            return feature

        query_feature = extract_feature(query_image_path)
        similarities = cosine_similarity([query_feature], features).flatten()

        results = [(image_paths[i], similarities[i]) for i in range(len(similarities)) if similarities[i] > 0.1]
        results = sorted(results, key=lambda x: x[1], reverse=True)

        if not results:
            st.warning("No results found with similarity > 0.1.")
        else:
            st.write(f"Top {min(len(results), 10)} results:")
            display_results(results, max_images=10)