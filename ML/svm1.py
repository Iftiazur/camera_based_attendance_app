import face_recognition
from sklearn import svm
import os
import cv2
import joblib
import tensorflow as tf

def train_face_recognizer(data_folder):
    # Training the SVC classifier
    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Loop through each person in the training directory
    for person in os.listdir(data_folder):
        person_folder = os.path.join(data_folder, person)

        # Loop through each training image for the current person
        for person_img in os.listdir(person_folder):
            image_path = os.path.join(person_folder, person_img)

            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(image_path)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If the training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for the current image with the corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(f"{person}/{person_img} is probably not a face")

    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale', verbose=True, max_iter=500)
    clf.fit(encodings, names)

    # Save the trained classifier to a file
    model_filename = "newFaceRecognition.pkl"
    joblib.dump(clf, model_filename)

    return clf

def test_face_recognizer(classifier, test_image_path):
    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(test_image_path)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)

    if not face_locations:
        print("No faces found in the test image.")
        return [], 0

    no = len(face_locations)
    print("Number of faces detected: ", no)

    # Get face encodings for all faces in the test image
    face_encodings = face_recognition.face_encodings(test_image)

    # Predict all the faces in the test image using the trained classifier
    recognized_faces = []
    for i in range(no):
        if i < len(face_encodings):
            test_image_enc = face_encodings[i]
            name = classifier.predict([test_image_enc])
            recognized_faces.append(name[0])

            # Draw a rectangle around the face
            top, right, bottom, left = face_locations[i]
            cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Add label text at the bottom of the rectangle
            label_text = f"Label: {name[0]}"
            cv2.putText(test_image, label_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Resize the output image to 1920x1080
    resized_image = cv2.resize(test_image, (1920, 1080))

    # Display the resized test image with rectangles and labels
    cv2.imshow('Resized Test Image with Recognized Faces', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return recognized_faces, no

def main():
    # Provide the path to the training data folder
    train_data_folder = "MTCNN_Face_Dataset" #MTCNN 

    # train_data_folder = "Haar_Face_Dataset" #Haar cascade

    # Provide the path to the test image
    test_image_path = "test.jpg"

    # Train the face recognizer
    face_classifier = train_face_recognizer(train_data_folder)

    # Test the face recognizer
    recognized_faces, num_faces_detected = test_face_recognizer(face_classifier, test_image_path)
    print("Recognized Faces:", recognized_faces)
    print("Number of Faces Detected:", num_faces_detected)

if __name__ == "__main__":
    main()
