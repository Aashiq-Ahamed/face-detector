from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from flask_cors import CORS  # Import CORS from flask_cors module
from sigDetect2 import detect_and_crop_signature

app = Flask(__name__)
CORS(app)  # Apply CORS to your Flask app


# Cropped Face API
@app.route('/process_image', methods=['GET'])
def process_image():
    image_path = request.args.get('image_path')
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400
    
    output_directory = os.path.join(os.path.dirname(image_path), "outputs")
    output_path = detect_faces_and_crop(image_path, output_directory)
    
    if output_path:
        return jsonify({"message": "success", "output_path": output_path}), 200
    else:
        return jsonify({"error": "Failed"}), 500
    

# Crop signature
@app.route('/process_sign', methods=['POST'])
def process_sign():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided in the request"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400
    
    # Call detect_and_crop_signature with the image file object
    result, base64_with_header = detect_and_crop_signature(image_file)
    
    if base64_with_header:
        return jsonify({"message": result, "cropped_image": base64_with_header}), 200
    else:
        return jsonify({"error HE HE": result}), 500



if __name__ == '__main__':
    app.run(host='localhost', port=5000)