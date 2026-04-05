import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer_for_efficientnet(model):
    """
    Returns the target layer for GradCAM.
    For timm's EfficientNet-B0, a good target is the last block.
    """
    if hasattr(model, 'conv_head'):
        return [model.conv_head]
    elif hasattr(model, 'blocks'):
        return [model.blocks[-1]]
    return None

def generate_gradcam(model, image_tensor, original_image):
    """
    Generates a Grad-CAM heatmap overlay for the input image.
    Returns the visualization (RGB array) and the grayscale_cam (2D array).
    """
    target_layers = get_target_layer_for_efficientnet(model)
    if target_layers is None or target_layers[0] is None:
        return None, None
        
    # Initialize GradCAM++ for richer multi-lesion localization
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]
    
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image.resize((224, 224)))
        
    rgb_img = np.float32(original_image) / 255.0
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization, grayscale_cam
