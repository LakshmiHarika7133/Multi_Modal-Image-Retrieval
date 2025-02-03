# Multi_Modal-Image-Retrieval
Using jupyter Notebook and Visual Studio Code

<img src="https://socialify.git.ci/your-github-username/Multi-Modal-Image-Retrieval/image?description=1&font=Source%20Code%20Pro&forks=1&issues=1&language=1&name=1&owner=1&pattern=Charlie%20Brown&pulls=1&stargazers=1&theme=Dark" alt="Multi-Modal-Image-Retrieval" width="1280" height="320" />

This project provides two functionalities:

### Text-to-Image Retrieval
Retrieve the top 10 most similar images based on a textual query.

### Image-to-Image Retrieval
Retrieve the top 10 most similar images by uploading an image.

## Project Overview

In modern image search and retrieval applications, multimodal image retrieval plays a crucial role in enhancing user experience. This project leverages machine learning techniques to perform both text-to-image and image-to-image retrieval efficiently.

## Dataset

We use a structured dataset containing images and associated metadata such as textual descriptions and labels. The dataset consists of:
- Images stored in the `images/` directory
- Metadata stored in `results.csv` containing `image_name` and `caption` for text-based retrieval
- link: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

## System Requirements

### Hardware:
1. 4GB RAM
2. i3 Processor
3. 1GB free space

### Software:
1. Visual Studio Code
2. Jupyter notebook
3. Python (>=3.8)

## Dependencies


Ensure you have the following libraries installed before running the project:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision streamlit tqdm pillow
```


## Usage

git clone https://github.com/your-github-username/Multi-Modal-Image-Retrieval.git
cd Multi-Modal-Image-Retrieval
python feature_extraction.py
streamlit run Multi_Modal.py

Open the URL displayed in the terminal to use the retrieval system.

## Results

We evaluate the performance of retrieval using cosine similarity scores. Results are displayed in the Streamlit interface, showcasing the top 10 similar images for both text and image queries.

## Model Deployment

Once satisfied with retrieval performance, the system can be deployed as an interactive web application or integrated into larger multimodal search platforms.

## Project Screenshots

Text-to-Image Retrieval Result
<img src="https://socialify.git.ci/your-github-username/Multi-Modal-Image-Retrieval/image?description=1&font=Source%20Code%20Pro&forks=1&issues=1&language=1&name=1&owner=1&pattern=Charlie%20Brown&pulls=1&stargazers=1&theme=Dark" alt="Multi-Modal-Image-Retrieval" width="1280" height="320" />


Image-to-Image Retrieval Result

<img src="https://socialify.git.ci/your-github-username/Multi-Modal-Image-Retrieval/image?description=1&font=Source%20Code%20Pro&forks=1&issues=1&language=1&name=1&owner=1&pattern=Charlie%20Brown&pulls=1&stargazers=1&theme=Dark" alt="Multi-Modal-Image-Retrieval" width="1280" height="320" />


## Overview:
This project provides two functionalities:
1.	Text-to-Image Retrieval: Retrieve the top 10 most similar images based on a textual query.
2.	Image-to-Image Retrieval: Retrieve the top 10 most similar images by uploading an image.
## Prerequisites:
1.	Ensure the following are installed and available on your system:
2.	Python (>=3.8)
3.	Required Python libraries (install them using the command below):
4.	pip install numpy pandas matplotlib scikit-learn torch torchvision streamlit tqdm pillow
5.	A dataset containing images and metadata (images/ folder and results.csv).
6.	Preferred development tools: Jupyter Notebook and Visual Studio Code.
## Open Jupyter Notebook and run the scripts:
1.	For Text-to-Image Retrieval: Run text_to_image.ipynb to preprocess text and vectorize captions using TF-IDF.
2.	For Image-to-Image Retrieval: Run image_to_image.ipynb to extract image features using a pre-trained ResNet-50 model.
## Organize the Dataset:
1.	Place all images in a folder named images inside the project directory.
2.	Ensure the results.csv file is in the project directory. This file should contain metadata like image_name and caption.

## How to Test the Project:
Text-to-Image Retrieval
1.	Open the text_to_image.ipynb file in Jupyter notebook.
2.	Input a query, e.g., "A man riding a bike in the mountains."
3.	Check the displayed results for the top 10 most similar images along with captions.
Image-to-Image Retrieval
1.	Open the image_to_image.ipynb file in Jupyter notebook.
2.	Upload a sample image (e.g., images/10002456.jpg).
3.	View the top 10 similar images displayed along with similarity scores.
## Test Retrieval Scripts:
1.	Open the corresponding Python scripts or notebooks.
2.	Provide either a textual query or an image as input.
3.	The system will return the top 10 similar images based on cosine similarity.
## How to Run Project:
1.	This project is submitted as a zip file. Follow the steps below to set up and run the project:
2.	Extract the zip file to a preferred location on your system.
3.	Open Visual Studio Code (VS Code) and import the extracted project folder, named Multi_Modal.
4.	Open the terminal in VS Code and install the necessary libraries by running the following command:
5.	pip install numpy pandas matplotlib scikit-learn torch torchvision streamlit tqdm pillow
6.	In the terminal, navigate to the project directory and execute the feature extraction script:
7.	python feature_extraction.py
8.	Start the web interface by running the following command in the terminal:
9.	streamlit run Multi_Modal.py
10.	After running the command, a URL will be provided in the terminal.
11.	Open the URL in a web browser to use the multimodal retrieval system.
## Launch the Streamlit Application:
1.	streamlit run app.py
2.	Navigate to the provided URL in your browser.
3.	Use the interface to input text queries or upload images.
## Project Folder Structure:

```bash
multimodal-image-retrieval/
├── Multi_Modal.py                      # Streamlit application for running the multimodal retrieval system
├── image_to_image.ipynb                # Jupyter notebook for implementing image-to-image retrieval
├── text_to_image.ipynb                 # Jupyter notebook for implementing text-to-image retrieval
├──Data_Preprocessing.ipynb                 # Jupyter notebook for data preprocessing and handling missing values
├── results.csv                         # Metadata file containing image and caption data for training
├── cleaned_results.csv                 # Preprocessed version of results.csv with cleaned data
├── features.npy                        # Numpy file storing precomputed feature vectors for images
├── images/                             # Folder containing all images used in the project
├── uploads/                            # Folder for storing uploaded images for queries
├── feature_extraction.py               # Python script for extracting image features using a pre-      trained model
├── Multi_Modal_User_Document           # User documentation describing project setup and usage instructions
├── Multi_Modal.ppt                     # PowerPoint presentation outlining the project
├── Multi_Modal_Final_Report            # Final project report with all details and findings
├── Multi_Modal_Dataset	 	# Dataset links and overview of Data
```

Error Handling and Debugging:
1.	Missing Dependencies: If a library is not found, install it using pip install <library-name>.
2.	File Not Found Errors: Ensure images/ and results.csv are correctly placed in the project directory.
3.	Low Accuracy: Check if the dataset is preprocessed correctly (e.g., clean captions, properly formatted metadata).
Future Improvements:
1.	Enhance the web interface for better user experience.
2.	Add support for real-time query processing.
3.	Optimize the retrieval models for faster performance.
