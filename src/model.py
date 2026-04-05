import torch
import torch.nn as nn
import timm

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
