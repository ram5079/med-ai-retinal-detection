import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer_for_efficientnet(model):
    """
    Returns the target layer for GradCAM.
    For timm's EfficientNet-B0, a good target is the last convolutional layer.
    Usually `model.conv_head` or the last block.
    """
    # timm's efficientnet_b0 has a 'conv_head' layer which is the last conv layer before pooling and classifier
    if hasattr(model, 'conv_head'):
        return [model.conv_head]
    # Fallback to the last block if conv_head is not found
    elif hasattr(model, 'blocks'):
        return [model.blocks[-1]]
    return None

def generate_gradcam(model, image_tensor, original_image):
    """
    Generates a Grad-CAM heatmap overlay for the input image.
    """
    target_layers = get_target_layer_for_efficientnet(model)
    if target_layers is None or target_layers[0] is None:
        return None
        
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate the heatmap mask
    # We disable use_cuda internally because we already placed model/tensor on device
    # pytorch-grad-cam handles the device mostly fine.
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]
    
    # Resize original image to 224x224 to match the expected overlay size
    # Grad-CAM mask is also 224x224
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image.resize((224, 224)))
        
    # Scale pixels to [0, 1] for show_cam_on_image
    rgb_img = np.float32(original_image) / 255.0
    
    # Create the overlay
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization, grayscale_cam

def generate_dynamic_explanation(grayscale_cam, predicted_class, class_names):
    """
    Generates a text explanation based on the regions activated in the Grad-CAM heatmap.
    """
    if grayscale_cam is None:
        return "Explanation could not be generated due to missing attention map."
        
    # Analyze heatmap
    # Grayscale cam is 224x224, values between [0, 1]
    
    height, width = grayscale_cam.shape
    
    # Define regions
    center_y, center_x = height // 2, width // 2
    radius = 50 # Region around the center (roughly macular area in standard framing)
    
    # Masks for different regions
    y, x = np.ogrid[:height, :width]
    center_mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
    edge_mask = ~center_mask
    
    # Focus intensities
    center_intensity = np.mean(grayscale_cam[center_mask])
    edge_intensity = np.mean(grayscale_cam[edge_mask])
    overall_max = np.max(grayscale_cam)
    
    # Base pattern identification
    high_activation_mask = grayscale_cam > 0.6
    num_hot_spots = 0
    
    # A bit of connected components to find distinct spots
    if np.any(high_activation_mask):
        import cv2
        num_hot_spots, _ = cv2.connectedComponents(high_activation_mask.astype(np.uint8))
        num_hot_spots -= 1 # subtract background
    
    # Generate textual explanation depending on class and heatmap heuristics
    stage = class_names[predicted_class]
    
    explanation = f"Model Prediction:  **{stage}**.\\n\\n"
    explanation += "**Heatmap Analysis:**\\n"
    
    if stage == "No DR":
        explanation += "The model found no significant pathological regions. The attention map is diffuse, which is expected for a healthy retina."
    else:
        if num_hot_spots > 3:
            explanation += "The model focused on multiple distinct spots. "
            if predicted_class >= 3:
                explanation += "These widespread activations suggest significant abnormalities like hemorrhages or hard exudates, typical of severe/proliferative DR. "
            else:
                explanation += "These activations might correlate with microaneurysms or early signs of vascular issues. "
        elif overall_max > 0.7:
            explanation += "There are highly concentrated regions of activation. "
            if center_intensity > edge_intensity:
                explanation += "The focus is particularly around the central retina (macular region), which is critical as it indicates potential central vision threat. "
            else:
                explanation += "The focus is towards the vascular arcades or peripheral regions, which might contain abnormal blood vessels or exudates. "
        else:
             explanation += "The model picked up some diffuse abnormal patterns hinting towards Diabetic Retinopathy, but without highly localized single spots."
             
    explanation += "\\n\\n*Note: This explanation is dynamically generated based on model attention patterns and should be interpreted alongside clinical judgment.*"
    return explanation

