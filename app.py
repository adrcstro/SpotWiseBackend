from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tempfile
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load model and handle potential loading errors
try:
    MODEL_PATH = 'densenet121.h5'
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

# Define class names and confidence threshold
CLASS_NAMES = [
    'Acne',
    'ChickenPox',
    'Melanoma',
    'Psoriasis',
    'Shingles',
    'Vitiligo',
    'Warts'
]
CONFIDENCE_THRESHOLD = 70  # Set the threshold to classify as "Unknown" if confidence is below this

@app.route('/', methods=['GET'])
def home():
    """
    Health-check endpoint to verify server status.
    """
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the skin disease from the uploaded image.
    If confidence is below the threshold, return "Unknown".
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        # Preprocess the image
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions using the model
        predictions = model.predict(img_array).flatten()
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions)) * 100

        # Remove the temporary file
        os.remove(temp_path)

        # Check if the confidence is below the threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Unknown prediction: confidence={confidence}, file={filename}")
            return jsonify({
                'class': 'Unknown',
                'confidence': f"{confidence:.2f}%",
                'message': 'This image does not match any known disease in our database.',
                'probabilities': {
                    CLASS_NAMES[i]: float(predictions[i]) * 100
                    for i in range(len(CLASS_NAMES))
                }
            })

        # Normal response for known classes
        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'probabilities': {
                CLASS_NAMES[i]: float(predictions[i]) * 100
                for i in range(len(CLASS_NAMES))
            }
        })
    except Exception as e:
        logger.error(f"Error processing the image: {e}")
        return jsonify({'error': f'Error processing the image: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0')
