import cv2
import os
import matplotlib.pyplot as plt

def detect_faces_and_crop(image_path, output_dir):
    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return "No faces found in the image."
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over detected faces and save them as separate images
    for i, (x, y, w, h) in enumerate(faces):
        cropped_face = image[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, f"cropped_face_{i+1}.jpg")
        cv2.imwrite(output_path, cropped_face)
    
    return f"Detected and cropped {len(faces)} faces. Saved in {output_dir}"

# Example usage
input_image_path = r'C:\Users\aashiq.a\Desktop\inputs\face1.png'
output_directory = r'C:\\Users\\aashiq.a\\Desktop\\outputs'
result = detect_faces_and_crop(input_image_path, output_directory)
outputPath = detect_faces_and_crop(input_image_path, output_directory)
print(result)

