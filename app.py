import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import logging
import time
import os

# --- Configuration and Setup ---

# 1. FIX: Use '__name__' instead of '_name_'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

MODEL_PATH = 'final_efficientnet_model.h5'
TARGET_SIZE = (224, 224)

# Define mock class labels for the example bounding boxes
MOCK_CLASSES = ['Cat', 'Dog', 'Chair', 'Bottle']

# --- Model Loading and Mocking ---

@st.cache_resource
def load_model():
    """
    Loads the TensorFlow/Keras model using st.cache_resource.
    Handles potential file errors by using a placeholder model if the file is not found.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}. Using mock predictions.")
            return None # Indicates the model failed to load
            
        st.info("Loading EfficientNet model...")
        # Note: tf.keras.models.load_model may require specific custom objects 
        # if your model uses custom layers, metrics, or loss functions.
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Using mock predictions.")
        logger.error(f"Model loading failed: {e}")
        return None


def mock_predict(preprocessed_image):
    """
    Mock function to simulate the structure of a model's object detection output.
    Returns example bounding boxes, classes, and scores.
    """
    # Create mock predictions that match the expected structure in draw_boxes
    mock_boxes = np.array([
        [20, 30, 100, 120],  # Mock Box 1 (Scaled for 224x224 input)
        [150, 150, 200, 210] # Mock Box 2
    ], dtype=np.float32)

    mock_classes = [MOCK_CLASSES[0], MOCK_CLASSES[1]]
    mock_scores = np.array([0.95, 0.88], dtype=np.float32)

    # Note: Keras model predictions are typically a list or tuple, 
    # so we mock the dictionary structure used by the user's 'draw_boxes' function.
    return {
        'boxes': mock_boxes,
        'classes': mock_classes,
        'scores': mock_scores
    }

# Attempt to load the real model
model = load_model()

# --- Core Functions ---

# Preprocessing function
def preprocess_image(image):
    """Resizes, converts to numpy array, normalizes, and adds batch dimension."""
    image_resized = image.resize(TARGET_SIZE)
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image.size  # Return original size for scaling boxes

# Function to draw bounding boxes and labels
def draw_boxes(image, predictions, original_size):
    """Draws bounding boxes and labels on the image based on model predictions."""
    draw = ImageDraw.Draw(image)
    
    # Attempt to load a better font, fall back to default
    try:
        # Use a system-provided font or a downloaded one (arial.ttf might not be portable)
        # Using a more robust system font path if available, or just load default
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # Ensure the prediction keys exist
    if not all(k in predictions for k in ['boxes', 'classes', 'scores']):
        logger.warning("Predictions missing required keys. Skipping box drawing.")
        st.warning("Prediction format is unexpected. Could not draw boxes.")
        return image

    boxes = predictions['boxes']
    classes = predictions['classes']
    scores = predictions['scores']

    # Scale factor for boxes (assuming boxes are normalized for TARGET_SIZE)
    scale_x = original_size[0] / TARGET_SIZE[0]
    scale_y = original_size[1] / TARGET_SIZE[1]

    for box, cls, score in zip(boxes, classes, scores):
        # 2. FIX: Ensure numpy arrays are converted to float/int before arithmetic operations
        if float(score) > 0.5:  # Threshold for displaying
            # Box format assumed: xmin, ymin, xmax, ymax (normalized)
            x1, y1, x2, y2 = box
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # Add a small offset to the label position
            draw.text((x1 + 5, y1 - 25), f"{cls}: {score:.2f}", fill="red", font=font)

    return image

# Function to log predictions
def log_predictions(predictions):
    """Logs the detected objects to the console."""
    logger.info("Predictions:")
    
    # Check if predictions are mocked/empty
    if not predictions or 'scores' not in predictions:
         logger.info("No predictions to log.")
         return

    for i, (box, cls, score) in enumerate(zip(predictions['boxes'], predictions['classes'], predictions['scores'])):
        if score > 0.5:
             logger.info(f"Object {i+1}: Class={cls}, Score={score:.2f}, Box={box}")

# --- Streamlit App UI ---

st.title("🖼 Image Upload and Object Detection App")
st.markdown(f"**Model Status:** {'✅ Real Model Active' if model else '⚠️ Mock Predictions Active (Model not found)'}")
st.markdown("""
Welcome! Upload an image to see object detection results using a (simulated) EfficientNet model.
""")

# Apply custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    /* Main container styling for the app's central block */
    .css-1d391kg, .css-1dp5qgq { 
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 5px;
        padding: 10px;
    }
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeaa7 !important;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display the original image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create two columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Uploaded Image')
            st.image(image, use_column_width=True)

        # Preprocess
        preprocessed, original_size = preprocess_image(image)

        # Run inference (Real or Mock)
        start_time = time.time()
        
        if model:
            # Use real model prediction
            predictions = model.predict(preprocessed)
            # You might need post-processing here to convert raw model output 
            # (e.g., from an SSD or YOLO model) into the expected dictionary format:
            # predictions = post_process_tf_output(predictions, class_names=MOCK_CLASSES)
        else:
            # Use mock prediction if model failed to load
            predictions = mock_predict(preprocessed)

        end_time = time.time()
        inference_time = end_time - start_time
        
        st.write(f"**Inference time:** `{inference_time:.2f}` seconds")

        if inference_time > 5 and model:
            st.warning("Inference took longer than 5 seconds. Consider optimizing the model or using a GPU.")
        
        # Draw boxes
        annotated_image = draw_boxes(image.copy(), predictions, original_size)

        with col2:
            st.subheader('Annotated Image')
            st.image(annotated_image, use_column_width=True, caption='Bounding boxes and labels applied.')
            
        # Log predictions to the console
        log_predictions(predictions)

        st.success("Inference completed successfully!")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logger.error(f"Processing error: {e}")
