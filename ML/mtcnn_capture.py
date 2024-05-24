import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from mtcnn import MTCNN


def capture_images():
    name = simpledialog.askstring("Input", "Enter your name:")
    if name is None:
        print("Name input canceled.")
        return 0  # Return 0 images captured
    roll = simpledialog.askinteger("Input", "Enter your roll number:")
    if roll is None:
        print("Roll number input canceled.")
        return 0  # Return 0 images captured

    path = "MTCNN_Face_Dataset/" + name + " - " + str(roll)
    num_of_images = 0
    total_images_to_capture = 50  # Set the total number of images to capture
    detector = MTCNN()
    try:
        os.makedirs(path)
    except FileExistsError:
        print('Directory Already Created')

    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        faces = detector.detect_faces(img)
        key = 0
        for result in faces:
            x, y, w, h = result['box']
            cv2.rectangle(img, (x, y), (x + w, y + h), (6, 2, 245), 2)
            new_img = img[y:y + h, x:x + w]  # Capture the face region

            # Display the progress meter
            meter_text = f"Captured: {num_of_images}/{total_images_to_capture}"
            cv2.putText(img, meter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (237, 19, 168), 2)

            # Display the "Press q to quit" message
            quit_text = "Press 'q' to quit"
            cv2.putText(img, quit_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("FaceDetection", img)
            key = cv2.waitKey(1) & 0xFF

            try:
                cv2.imwrite(f"{path}/{num_of_images}_{name}.jpg", new_img)
                num_of_images += 1
            except Exception as e:
                print(f"Error saving image: {e}")

        if key == ord("q") or key == 27 or num_of_images >= total_images_to_capture:
            break

    vid.release()
    cv2.destroyAllWindows()
    return num_of_images


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    while True:
        num_of_images = capture_images()
        if num_of_images >= 30:
            print("Images captured successfully.")
        else:
            print("Failed to capture required number of images.")

        choice = messagebox.askquestion("Capture More Images", "Do you want to capture more images?")
        if choice == 'no':
            break


if __name__ == "__main__":
    main()
