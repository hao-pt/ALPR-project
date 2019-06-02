import cv2
import numpy as np
from PIL import Image, ImageFilter
from pytesseract import image_to_string
import os
import xml.etree.ElementTree as ET

def classify_plate(plate_img):
    tree = ET.parse(r'E:\K16\Junior\TGMT\ALPR-project\SVM.xml')  
    root = tree.getroot()

    # Read model parametrers and labels
    SVM_TrainingData = root.find('TrainingData')
    SVM_Classes = root.find('classes')
    
    training_data = SVM_TrainingData.find("data").text
    labels = SVM_Classes.find("data").text

    labels = labels.split(sep = " ")
    rows = []
    for c in labels:
        if c == '\n' or c == '':
            continue
        if len(c) >= 2:
            if '\n' in c:
                num = int(c[:-1])  
        else:
            num = int(c)
            
        rows.append(num)

    labels = np.array(rows, dtype = np.int32).reshape(-1, 1)

    training_data = training_data.split(" ")
    rows2 = []
    for c in training_data:
        if c == '\n' or c == '':
            continue
        if '\n' in c:
            num = float(c[:-1])
        else:
            num = float(c)
            
        rows2.append(num)
    
    h, w = int(SVM_TrainingData.find("rows").text), int(SVM_TrainingData.find("cols").text)
    training_data = np.array(rows2, dtype = np.float32).reshape((h, w))

    # Set SVM params
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.01))
    svm.setDegree(0)
    svm.setGamma(1)
    svm.setCoef0(0)
    svm.setC(1)
    svm.setNu(0)
    svm.setP(0)
    
    print(training_data.shape, labels.shape)
    # Train model
    svm.train(training_data, cv2.ml.ROW_SAMPLE, labels)

    # Predict
    p = plate_img.reshape(1, -1).astype(np.float32)

    response = svm.predict(p)[1]

    if response == 1:
        return True
    return False



# Load cascade model
cascade = cv2.CascadeClassifier(r"E:\K16\Junior\TGMT\ALPR-project\haarcascade_russian_plate_number.xml") # haarcascade_licence_plate_rus_16stages.xml

# # Test all file in dir
# path = "E:\\K16\\Junior\\TGMT\\ALPR-project\\test_images\\"
# count = 0

# # root, directories, files
# for root, directories, files in os.walk(path):
#     for f in files:
#         if "jpg" in f or "JPG" in f:
#             # Read image
#             img = cv2.imread(path + f)

#             # Resize image
#             width = 600
#             height = int(600 * img.shape[0] / img.shape[1])
#             img = cv2.resize(img, (width, height))

#             # Convert to gray image
#             grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#             # Equalize brightness and increase the contrast
#             grayImg = cv2.equalizeHist(grayImg)

#             # detect license plate
#             box_plates = cascade.detectMultiScale(grayImg) #, scaleFactor = 1.1, minNeighbors = 3, flags = cv2.CASCADE_SCALE_IMAGE)

#             # Draw bounding box on detected plate
#             for (x, y, w, h) in box_plates:
#                 # Draw bounding box
#                 img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 # # Plate roi
#                 # plate_roi = np.copy(img[y:y+h, x:x + w])
#                 # cv2.imshow("Plate ROI", plate_roi)
#                 # OCR
#                 # character_recognition(plate_roi, (x, y, w, h), img)

#             cv2.imshow("Plate detection " + str(count), img)
#             count += 1

# cv2.waitKey(0)
# cv2.destroyAllWindows()

    

# Load image
img = cv2.imread(r"E:\K16\Junior\TGMT\ALPR-project\test_images\9588DWV.jpg")

# Resize image
width = 600
height = int(600 * img.shape[0] / img.shape[1])
img = cv2.resize(img, (width, height))

# Convert to gray image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Equalize brightness and increase the contrast
grayImg = cv2.equalizeHist(grayImg)

# detect license plate
box_plates = cascade.detectMultiScale(grayImg) #, scaleFactor = 1.1, minNeighbors = 3, flags = cv2.CASCADE_SCALE_IMAGE)

# Draw bounding box on detected plate
for (x, y, w, h) in box_plates:
    # Draw bounding box
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Plate roi
    plate_roi = np.copy(img[y:y+h, x:x + w])
    cv2.imshow("Plate ROI", plate_roi)

    # OCR
    # character_recognition(plate_roi, (x, y, w, h), img)

# classify_plate()

cv2.imshow("Plate detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()