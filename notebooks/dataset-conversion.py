import cv2
import numpy as np
import os
from pathlib import Path

def mask_to_yolo_polygons(mask_path, output_txt_path, color_to_class_map):
    """
    Converts a single color mask image to YOLO segmentation format.
    
    :param mask_path: Path to the mask image
    :param output_txt_path: Path where the output .txt file will be saved
    :param color_to_class_map: Dict mapping BGR tuple colors to integer class IDs
    """
    # Load mask image in color
    mask = cv2.imread(str(mask_path))
    if mask is None:
        print(f"Error loading mask: {mask_path}")
        return
        
    height, width, _ = mask.shape
    
    # Clean/create output file
    with open(output_txt_path, 'w') as f:
        pass

    # Process each class color mapping
    for color, class_id in color_to_class_map.items():
        # OpenCv uses BGR, ensure your tuple matches BGR order
        bgr_color = np.array(color, dtype=np.uint8)
        
        # Create a binary mask isolating this specific color/class
        class_mask = cv2.inRange(mask, bgr_color, bgr_color)
        
        # Find contours (boundaries) of the isolated class
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with open(output_txt_path, 'a') as f:
            for contour in contours:
                # Flatten the contour array to [x1, y1, x2, y2, ...]
                contour = contour.reshape(-1, 2)
                
                # YOLO requires at least 3 points to create a polygon (triangle)
                if len(contour) < 3:
                    continue
                
                # Normalize the coordinates
                normalized_coords = []
                for point in contour:
                    x_norm = point[0] / width
                    y_norm = point[1] / height
                    normalized_coords.append(f"{x_norm:.6f} {y_norm:.6f}")
                
                # Construct the line: class_id followed by coordinates
                line = f"{class_id} " + " ".join(normalized_coords) + "\n"
                f.write(line)

# --- Example Usage ---
# Map your unique mask colors (BGR format) to class integer IDs
# Example: Blue mask area -> Class 0, Green mask area -> Class 1
color_map = {
    (255, 0, 0): 0,    # Pure Blue in BGR
    (0, 255, 0): 1,    # Pure Green in BGR
}

# Convert a single mask
mask_to_yolo_polygons("mask_001.png", "img001.txt", color_map)