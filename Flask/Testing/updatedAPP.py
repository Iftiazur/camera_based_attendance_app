from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import joblib
import os

app = Flask(__name__)

# Load the trained face recognition model
face_classifier = joblib.load("updatedFaceRecognition.pkl")
known_face_names = joblib.load("label_encoder.pkl")

def recognize_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_locations:
        return [], 0, "No faces detected in the image", []

    recognized_faces = []
    for face_encoding in face_encodings:
        # Predict the face using the trained classifier
        name = face_classifier.predict([face_encoding])[0]
        recognized_faces.append(name)

    num_faces_detected = len(face_locations)
    return recognized_faces, num_faces_detected, "Faces detected and recognized successfully", face_locations

@app.route('/detect_faces', methods=['POST', 'GET'])
def detect_faces():
    if request.method == 'GET':
        return jsonify({"status": "GET method is working correctly"})

    if 'image' not in request.files:
        return jsonify({"status": "No image uploaded"}), 400

    image_file = request.files['image']

    # Save the image temporarily
    temp_image_path = "test.jpg"
    image_file.save(temp_image_path)

    # Detect and recognize faces
    recognized_faces, num_faces_detected, status_message, face_locations = recognize_faces(temp_image_path)

    # Remove the temporary image
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    # Prepare response data in JSON format
    response_data = {
        "status": status_message,
        "recognized_faces": recognized_faces,
        "num_faces_detected": num_faces_detected,
        "face_locations": face_locations
    }

    return jsonify(response_data)

@app.route('/trained_faces', methods=['GET'])
def trained_faces():
    return jsonify({"trained_faces": known_face_names})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
