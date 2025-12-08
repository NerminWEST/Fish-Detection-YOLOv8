from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('yolov8n.pt')

# Run detection on an image
results = model('test/images/FishDataset689_png.rf.5740179a69022d7b20d4641084c9839b.jpg')

# Display results
for result in results:
    result.show()  # Display image with detections
    result.save(filename='detected_fish.jpg')  # to  save result
