import numpy as np
import cv2
import json

def calculate_lesion_metrics(grayscale_cam, activation_threshold=0.6):
    """
    Analyzes the Grad-CAM continuous mask (0.0 to 1.0 array) to output a fixed scoring standard
    and extracting categorical logic for the LLM JSON template.
    """
    if grayscale_cam is None:
        return 0, 0.0, None, {}

    # 1. Base thresholds and extraction
    high_act_mask = grayscale_cam >= activation_threshold
    affected_pixels = np.sum(high_act_mask)
    total_pixels = grayscale_cam.size
    
    affected_percentage = (affected_pixels / total_pixels) * 100
    
    if affected_pixels == 0:
        return 0, 0.0, high_act_mask, {
            "intensity": "none", "spread": "none", "location": "none", "area_percent": 0.0
        }
        
    mean_intensity = np.mean(grayscale_cam[high_act_mask])
    
    # 2. Score Calculation
    area_factor = min(affected_percentage / 15.0, 1.0)
    intensity_factor = max(0.0, min((mean_intensity - activation_threshold) / (1.0 - activation_threshold), 1.0))
    
    raw_score = (area_factor * 0.7 + intensity_factor * 0.3) * 100
    if affected_percentage < 0.5 and mean_intensity < 0.7:
         raw_score = raw_score * 0.5
    final_score = int(np.clip(np.round(raw_score), 0, 100))
    
    # 3. Structural Attributes Mapping (V4 JSON Construction)
    cat_intensity = "high" if mean_intensity > 0.85 else "medium" if mean_intensity > 0.7 else "low"
    
    # Spread via components
    num_spots = 0
    if np.any(high_act_mask):
        num_spots, _ = cv2.connectedComponents(high_act_mask.astype(np.uint8))
        num_spots -= 1
        
    if affected_percentage > 10.0 or num_spots > 4:
        cat_spread = "diffuse"
    elif affected_percentage > 2.0 or num_spots > 1:
        cat_spread = "moderate"
    else:
        cat_spread = "localized"
        
    # Location tracking
    height, width = grayscale_cam.shape
    cy, cx = height // 2, width // 2
    radius = min(height, width) // 3
    y, x = np.ogrid[:height, :width]
    center_mask = ((x - cx)**2 + (y - cy)**2 <= radius**2)
    edge_mask = ~center_mask
    
    c_int = np.sum(grayscale_cam[center_mask & high_act_mask])
    e_int = np.sum(grayscale_cam[edge_mask & high_act_mask])
    
    if c_int > (e_int * 2):
        cat_location = "macula"
    elif e_int > (c_int * 2):
        cat_location = "peripheral"
    else:
        cat_location = "mixed"

    structured_json = {
        "intensity": cat_intensity,
        "spread": cat_spread,
        "location": cat_location,
        "area_percent": round(affected_percentage, 1)
    }
    
    return final_score, affected_percentage, high_act_mask, structured_json
