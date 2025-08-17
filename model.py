# filename: model.py

import torch
from torch import nn
from torchvision import models, transforms
import sys
from PIL import Image
import io

# --- CONFIGURATION ---
DEVICE = "cpu"
MODEL_PATH = "models/house_plant_classifier_v1.pth"
CLASS_NAMES_PATH = "class_names.txt"
NUM_CLASSES = 47 # The number of plant species you have

# --- MODEL AND TRANSFORMS ---
# Define the model architecture
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=NUM_CLASSES),
)

# Define the image transformations
# These transforms are from the default EfficientNet-B0 weights and are a good standard
auto_transforms = models.EfficientNet_B0_Weights.DEFAULT.transforms()


# --- UTILITY FUNCTIONS ---
def load_trained_model():
    """
    Loads the trained state dictionary into the model architecture.
    This function is called once when the API server starts.
    """
    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print("Model weights loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"--- FATAL ERROR ---")
        print(f"Model file not found at: '{MODEL_PATH}'")
        print("Please ensure the .pth file is in the 'models' directory.")
        sys.exit(1) # Exit if the model can't be found
    except Exception as e:
        print(f"--- FATAL ERROR ---")
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

def load_class_names():
    """
    Loads the class names from the text file.
    This function is called once when the API server starts.
    """
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        print("Class names loaded successfully.")
        return class_names
    except Exception as e:
        print(f"--- FATAL ERROR ---")
        print(f"Could not read class names file at '{CLASS_NAMES_PATH}': {e}")
        sys.exit(1)


# --- THIS IS THE UPDATED FUNCTION ---
def predict_image(model: nn.Module, class_names: list, image_bytes: bytes) -> tuple[str, float]:
    """
    Makes a prediction on a single image provided as bytes.
    Includes optimizations for faster inference.
    """
    # OPTIMIZATION 1: Disable gradient calculations
    torch.set_grad_enabled(False)
    
    # OPTIMIZATION 2: Ensure model is in evaluation mode
    model.eval()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Transform the image and add a batch dimension
    image_tensor = auto_transforms(image).unsqueeze(0).to(DEVICE)

    # OPTIMIZATION 3: Use the inference_mode context manager for efficiency
    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        
    # Get top prediction details
    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_class = class_names[predicted_idx.item()]
    
    # Return the class name and confidence as a percentage
    return predicted_class, confidence.item() * 100