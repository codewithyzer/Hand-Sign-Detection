import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
padding = 30
save_folder = "data/U"
img_size = 224
count = 0

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
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

    # Create white square background instead of black
    result = np.ones((size, size, 3), dtype=np.uint8) * 255
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
    return result

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
            cv2.imshow("ImageCropResized", imgResized)

            key = cv2.waitKey(1)
            if key == ord('s'):
                filename = os.path.join(save_folder, f"image_{count}.jpg")
                cv2.imwrite(filename, imgResized)
                print(f"[Saved] {filename}")
                count += 1
        else:
            cv2.waitKey(1)

    cv2.imshow("Webcam", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
