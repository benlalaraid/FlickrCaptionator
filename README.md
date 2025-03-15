# Flickr Caption AI - Modern Web Application

A modern, immersive web application that integrates a pre-trained Flickr caption generation model with a beautiful dark-themed UI.

## Features

- **Modern Dark UI**: Sleek, responsive design with interactive elements
- **Drag & Drop**: Easy image upload with drag and drop functionality
- **Interactive Particles**: Dynamic background with interactive particles
- **FastAPI Backend**: Robust Python backend for image processing and caption generation
- **Real-time Caption Generation**: Generate captions for any uploaded image

## Project Structure

```
flickr-caption-project/
├── app.py                  # FastAPI application
├── models/                 # Contains the trained model
│   └── best_model.h5       # Pre-trained caption generation model
├── notebooks/              # Original notebooks and Python files
├── static/                 # Static files
│   ├── css/                # Stylesheets
│   │   └── styles.css      # Main stylesheet
│   ├── js/                 # JavaScript files
│   │   └── main.js         # Main JavaScript file
│   └── images/             # Image assets
├── templates/              # HTML templates
│   └── index.html          # Main application page
└── requirements.txt        # Python dependencies
```

## Setup and Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
uvicorn app:app --reload
```

3. Open your browser and navigate to `http://localhost:8000`

## Model Information

The application uses a deep learning model trained on the Flickr8k dataset. The model architecture combines:

- **VGG16**: For image feature extraction (4096-dimensional feature vector)
- **LSTM**: For sequence processing and caption generation
- **Embedding Layer**: For word representation
- **Dense Layers**: For feature fusion and word prediction

The model takes an image as input and generates a descriptive caption based on the content of the image.

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, Particles.js
- **Backend**: Python, FastAPI
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: Pillow, VGG16

## Usage

1. Upload an image by dragging and dropping or using the file selector
2. Click "Generate Caption" to process the image
3. View the generated caption
4. Copy or share the caption as needed


![image alt](https://github.com/benlalaraid/FlickrCaptionator/blob/main/SharedScreenshot.jpg?raw=true)
