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
nms_threshold = 0.4  # Non-maxima suppression threshold

# Load YOLO model (use yolov4-tiny for better performance)
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

if use_gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load class names
with open('coco.names', 'r') as f:
    CLASSES = f.read().strip().split('\n')

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Function to perform object detection on a frame
def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    try:
        output_layers_indices = net.getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]
    except IndexError as e:
        print(f"Error in fetching output layers: {e}")
        return frame

    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_level:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_level, nms_threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in np.random.randint(0, 255, size=3)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(CLASSES[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# Function to open an image file
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = imutils.resize(image, width=640)  # Reduce width for less RAM usage
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
            
            frame = imutils.resize(frame, width=640)  # Reduce width for less RAM usage
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
        frame = imutils.resize(frame, width=640)  # Reduce width for less RAM usage
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
