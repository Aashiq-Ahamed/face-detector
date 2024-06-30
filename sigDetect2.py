import cv2
from skimage import measure
import base64
import numpy as np

def detect_and_crop_signature(original_img_path):
    # Read the original image
    original_img = cv2.imread(original_img_path)
    
    if original_img is None:
        return "Failed to read the input image.", None

    # Convert the image to grayscale
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Parameters to adjust based on your image characteristics
    SMALL_SIZE_THRESHOLD = 500  # Minimum area of a valid signature
    LARGE_SIZE_THRESHOLD = 5000  # Maximum area of a valid signature

    # Threshold the image to get a binary image
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Perform connected component analysis
    labels = measure.label(binary_img, connectivity=2)
    props = measure.regionprops(labels)

    # Find the largest valid component based on area
    valid_components = [prop for prop in props if SMALL_SIZE_THRESHOLD < prop.area < LARGE_SIZE_THRESHOLD]
    if not valid_components:
        return "No signature found in the image.", None
    else:
        largest_component = max(valid_components, key=lambda prop: prop.area)
        minr, minc, maxr, maxc = largest_component.bbox

        # Crop the original image
        cropped_signature = original_img[minr:maxr, minc:maxc]

        # Convert cropped image to base64
        retval, buffer = cv2.imencode('.png', cropped_signature)  # Ensure the correct format is used here
        if not retval:
            return "Error encoding image to base64.", None

        base64_cropped_image = base64.b64encode(buffer).decode('utf-8')
        base64_with_header = f"data:image/png;base64,{base64_cropped_image}"

        return "Signature cropped and converted to base64 with header.", base64_with_header
