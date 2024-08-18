import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from imutils.video import VideoStream, FPS
import numpy as np
import imutils
import cv2

# Set up configuration
use_gpu = True
confidence_level = 0.5
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "banana", "apple", "orange", "book"]  # Added more objects for demonstration
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the model
net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

if use_gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Function to perform object detection on a frame
def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_level:
            idx = int(detections[0, 0, i, 1])
            if idx < len(CLASSES):  # Check to avoid index error if more classes are added
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)
    
    return frame

# Function to open an image file
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = imutils.resize(image, width=1080, height=720)
        detected_image = detect_objects(image)
        save_image(detected_image)
        show_image(detected_image)

# Function to open a video file
def upload_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        vs = cv2.VideoCapture(file_path)
        fps = FPS().start()

        while True:
            ret, frame = vs.read()
            if not ret:
                break
            
            frame = imutils.resize(frame, width=1080)
            detected_frame = detect_objects(frame)
            cv2.imshow("Video Detection", detected_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

            fps.update()

        fps.stop()
        vs.release()
        cv2.destroyAllWindows()

# Function to start camera stream
def scan_using_camera():
    vs = VideoStream(src=0).start()
    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1080)
        detected_frame = detect_objects(frame)
        
        cv2.imshow("Camera Detection", detected_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            break

        fps.update()

    fps.stop()
    vs.stop()
    cv2.destroyAllWindows()

# Function to save the detected image
def save_image(image):
    filename = os.path.join('results', 'detected_{}.jpg'.format(len(os.listdir('results')) + 1))
    cv2.imwrite(filename, image)

# Function to show image using Tkinter
def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    
    panel = tk.Label(image=image)
    panel.image = image
    panel.pack()

# Setting up the GUI
root = tk.Tk()
root.title("Object Detection")
root.geometry("1080x720")

btn_upload_image = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload_image.pack(side="left", padx=10, pady=10)

btn_upload_video = tk.Button(root, text="Upload Video", command=upload_video)
btn_upload_video.pack(side="left", padx=10, pady=10)

btn_camera = tk.Button(root, text="Scan using Camera", command=scan_using_camera)
btn_camera.pack(side="right", padx=10, pady=10)

root.mainloop()
