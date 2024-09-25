import speech_recognition as sr
import spacy
import cv2
import numpy as np
from ultralytics import YOLO

recognizer = sr.Recognizer()
mic = sr.Microphone()
nlp = spacy.load('en_core_web_sm')


def recognize_speech():
    with mic as source:
        print("Say something!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"Recognized: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
        return None


def interpret_command(command):
    doc = nlp(command)
    action = None
    direction = None
    object = None
    for token in doc:
        if token.lemma_ in ["move", "go", "turn"]:
            action = token.lemma_
        if token.lemma_ in ["left", "right", "forward", "backward"]:
            direction = token.lemma_
        if token.lemma_ in ["cup", "bottle", "car", "banana"]:
            object = token.lemma_
    if action or direction or object:
        return direction, action, object
    else:
        return None, None, None

# def send_command():
#     command = recognize_speech()
#     if command:
#         action, direction, object = interpret_command(command)
#         objects = []
#         if action or direction or object:
#             objects.append(object)
#             print(objects)
#     return objects



# Load YOLO model files
model = YOLO('yolo-Weights/yolov8n.pt')

# Load the COCO dataset class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to get class IDs based on user input
def get_class_ids(object_names, classes):
    class_ids = []
    for obj in object_names:
        if obj in classes:
            class_ids.append(classes.index(obj))
        else:
            print(f"Object '{obj}' not found in classes.")
    return class_ids

# User input: list of objects to detect
# input_objects = ["cup", "apple"]  # You can replace this with any objects you want to detect
# object_class_ids = get_class_ids(send_command(), classes)

def detect_objects(id):
    while True:

        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model.predict(source=frame, classes=id, show=False)

        for result in results:
            boxes = result.boxes  # Get the bounding boxes
            for box in boxes:
            # box.xyxy gives [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = box.xyxy[0].numpy()  # Convert tensor to numpy array

            # Calculate width and height
                width = x_max - x_min
                height = y_max - y_min

            # Print the bounding box (x, y, width, height)
                print(f"x: {x_min}, y: {y_min}, width: {width}, height: {height}")

            # Optional: Draw bounding box on the frame
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, "Cup", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the resulting frame with detected cups
                cv2.imshow('YOLOv8 Cup Detection',frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cam = cv2.VideoCapture(0)

while True:

    command = recognize_speech()

    if command:
        action, direction, object = interpret_command(command)
        objects = []
        if action or direction or object:
            objects.append(object)

        object_class_ids = get_class_ids(objects, classes)
        detect_objects(object_class_ids)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break







cam.release(0)
cv2.destroyAllWindows()
