import torch
import torch.nn as nn
import timm
from torchvision import transforms
import numpy as np

# Function to load model
def load_model(model_path, device):
    """
    Loads the EfficientNet-B0 model with 5 output classes.
    """
    try:
        # Load the EfficientNet-B0 architecture using timm
        model = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Modify the classifier to output 5 classes
        model.classifier = nn.Linear(model.classifier.in_features, 5)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to get transformation pipeline
def get_transforms():
    """
    Returns the transformation pipeline for EfficientNet-B0.
    1. Resize to 224x224
    2. Convert to Tensor
    3. Normalize using ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to process an image and return prediction
def predict(model, image, device, transform=None):
    """
    Takes a PIL image, applies transformations, and runs it through the model.
    """
    if transform is None:
        transform = get_transforms()
        
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    probs_np = probabilities.cpu().numpy()
    predicted_class = np.argmax(probs_np)
    confidence = probs_np[predicted_class]
    
    return predicted_class, confidence, probs_np, image_tensor

