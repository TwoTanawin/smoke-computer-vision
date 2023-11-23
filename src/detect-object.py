import cv2
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('/app/smoke-detect/yolov8-weight/yolov8l.pt')

# Define path to the image file
source = '/app/smoke-detect/computer-vision/images/maxresdefault-3.jpg'

img = cv2.imread(source)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference on the source
results = model(source)  # list of Results objects

for result in results:
    boxes = result.boxes
# bbox = boxes.xyxy.tolist()[0]

for i in range(len(boxes)):
    bbox = boxes.xyxy.tolist()[i]
    x1, y1, x2, y2 = map(int, bbox)  # Converting to integers

    roi = image_rgb[int(y1):int(y2), int(x1):int(x2)]

    # plt.imshow(roi)
    
    # classes = names[int(r.boxes.cls[i])]


    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # print(classes)
    

    cv2.imwrite(f"/app/smoke-detect/computer-vision/output/img_{i}.jpg", roi_rgb)
    