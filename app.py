# filename: app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import model as model_predictor # Import our refactored model utility file

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
# This is crucial to allow your Vercel frontend to make requests to this API
CORS(app) 

# --- MODEL LOADING ---
# Load the model and class names once when the server starts
print("Loading model and class names...")
try:
    model = model_predictor.load_trained_model()
    class_names = model_predictor.load_class_names()
    print("Ready to make predictions.")
except Exception as e:
    # This will catch any exit signals from the model loader
    print(f"Application failed to start: {e}")
    model = None
    class_names = None

# --- API ROUTES ---

@app.route('/', methods=['GET'])
def health_check():
    """A simple health check endpoint to confirm the API is running."""
    return jsonify({
        "status": "success",
        "message": "API is running. Send a POST request to /predict to classify an image."
    })

@app.route('/predict', methods=['POST'])
def predict():
    """The main prediction endpoint."""
    if model is None or class_names is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503

    # 1. Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # 2. Check if the file is empty
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
            
    # 3. If the file is valid, proceed with prediction
    if file:
        try:
            # Read the image file as bytes
            image_bytes = file.read()
            
            # Make a prediction using our model utility
            predicted_class, confidence = model_predictor.predict_image(
                model=model,
                class_names=class_names,
                image_bytes=image_bytes
            )
            
            # 4. Return the result as a JSON response
            return jsonify({
                "status": "success",
                "prediction": {
                    "species": predicted_class,
                    "confidence": f"{confidence:.2f}"
                }
            })
        except Exception as e:
            print(f"Prediction Error: {e}")
            return jsonify({"status": "error", "message": "Failed to process image"}), 500
            
    return jsonify({"status": "error", "message": "Invalid request"}), 400


if __name__ == '__main__':
    # Using host='0.0.0.0' makes it accessible on your local network
    app.run(debug=True, host='0.0.0.0', port=5000)