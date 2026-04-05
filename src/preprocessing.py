import torch
from torchvision import transforms
import numpy as np

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

def predict(model, image, device, transform=None):
    """
    Takes a PIL image, applies transformations, and runs it through the model.
    """
    if transform is None:
        transform = get_transforms()
        
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply temperature scaling to soften overconfident logits
        temperature = 2.0
        scaled_logits = outputs[0] / temperature
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=0)
        
    probs_np = probabilities.cpu().numpy()
    predicted_class = int(np.argmax(probs_np))
    confidence = float(probs_np[predicted_class])
    
    return predicted_class, confidence, probs_np, image_tensor
