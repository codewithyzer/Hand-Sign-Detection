import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('model/keras_model.h5', 'model/labels.txt')
padding = 30
img_size = 224

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.readlines()
    return [label.strip() for label in labels]

def resize_with_padding(img, size=224):
    h, w = img.shape[:2]
    aspect_ratio = w / h

    if aspect_ratio > 1:
        new_w = size
        new_h = int(size / aspect_ratio)
    else:
        new_h = size
        new_w = int(size * aspect_ratio)

    img_resized = cv2.resize(img, (new_w, new_h))
    result = np.ones((size, size, 3), dtype=np.uint8) * 255
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
    return result

labels = load_labels('model/labels.txt')

while True:
    success, img = capture.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        height, width, _ = img.shape
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        if x2 > x1 and y2 > y1:
            imgCrop = img[y1:y2, x1:x2]
            imgResized = resize_with_padding(imgCrop, img_size)

            prediction, index = classifier.getPrediction(imgResized, draw=False)

            label_text = f"{labels[index]}: {prediction[index] * 100:.2f}%"

            label_x_position = x1
            label_y_position = y2 + 30
            label_y_position = min(label_y_position, height - 10)

            label_color = (255, 255, 255)
            cv2.putText(img, label_text, (label_x_position, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

            cv2.imshow("ImageCropResized", imgResized)

    cv2.imshow("Webcam", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
