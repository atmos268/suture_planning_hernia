import numpy as np
import cv2
def keep_largest_connected_component(imgPath):
    # Ensure the input image is in binary format (0 and 255).
    input_image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    # Ensure the image is binary (threshold if needed).
    _, binary_image = cv2.threshold(input_image, 128, 255, cv2.THRESH_BINARY)
    if len(np.unique(binary_image)) != 2:
        print(len(np.unique(binary_image)))
        raise ValueError("Input image must be binary (0 and 255).")

    # Find connected components in the binary image.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Find the index of the largest connected component (excluding the background).
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a binary mask for the largest component.
    largest_component_mask = (labels == largest_component_index).astype(np.uint8) * 255

    return largest_component_mask
