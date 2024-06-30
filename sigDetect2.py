import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology

# Parameters to adjust based on your image characteristics
SMALL_SIZE_THRESHOLD = 500  # Minimum area of a valid signature
LARGE_SIZE_THRESHOLD = 5000  # Maximum area of a valid signature

# Read the input image in grayscale
img = cv2.imread(r'C:\Users\aashiq.a\Desktop\download6.png', cv2.IMREAD_GRAYSCALE)
original_img = cv2.imread(r'C:\Users\aashiq.a\Desktop\download6.png')

# Threshold the image to get a binary image
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Perform connected component analysis
labels = measure.label(binary_img, connectivity=2)
props = measure.regionprops(labels)

# Find the largest valid component based on area
valid_components = [prop for prop in props if SMALL_SIZE_THRESHOLD < prop.area < LARGE_SIZE_THRESHOLD]
if not valid_components:
    print("No signature found in the image.")
else:
    largest_component = max(valid_components, key=lambda prop: prop.area)
    minr, minc, maxr, maxc = largest_component.bbox

    # Crop the original image
    cropped_signature = original_img[minr:maxr, minc:maxc]

    # Save the cropped signature
    output_path = r'C:\Users\aashiq.a\Desktop\outputs\cropped_signature.jpg'
    cv2.imwrite(output_path, cropped_signature)
    print(f"Signature cropped and saved as {output_path}")

    # Display the cropped signature (optional)
    plt.imshow(cv2.cvtColor(cropped_signature, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
