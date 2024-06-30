from flask import Flask, request, jsonify
import os
from face2 import detect_faces_and_crop
from sigDetect2 import detect_and_crop_signature

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

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
    

# Cropped Signaure API
@app.route('/process_sig', methods=['GET'])
def process_sig():
    image_path = request.args.get('image_path')
    output_directory = request.args.get('output_dir')
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, 'cropped_signature.jpg')
    
    result, saved_path = detect_and_crop_signature(image_path, output_path)
    
    if saved_path:
        return jsonify({"message": result, "output_path": saved_path}), 200
    else:
        return jsonify({"error": result}), 500


if __name__ == '__main__':
    app.run(host='localhost', port=5000)