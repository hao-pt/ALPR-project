import cv2
import numpy as np
from PIL import Image, ImageFilter
from pytesseract import image_to_string

# Load cascade model
cascade = cv2.CascadeClassifier("haarcascade_licence_plate_rus_16stages.xml")

# Load image
img = cv2.imread(r"E:\K16\Junior\TGMT\ALPR-project\test_images\3732FWW.JPG")

# Resize image
width = 600
height = int(600 * img.shape[0] / img.shape[1]) 
img = cv2.resize(img, (width, height))

# Convert to gray image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Equalize brightness and increase the contrast
grayImg = cv2.equalizeHist(grayImg)

# detect license plate
box_plates = cascade.detectMultiScale(grayImg, scaleFactor = 1.3, minNeighbors = 3, flags = cv2.CASCADE_SCALE_IMAGE)

# Draw bounding box on detected plate
for (x, y, w, h) in box_plates:
    # Draw bounding box
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Plate roi
    plate_roi = np.copy(img[y:y+h, x:x + w])
    cv2.imshow("Plate ROI", plate_roi)
    # OCR
    # character_recognition(plate_roi, (x, y, w, h), img)

cv2.imshow("Plate detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()