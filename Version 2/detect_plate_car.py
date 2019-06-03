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


# Use floodFill algorithm for more precise cropping
def floodFill(rects, plate_image):
    # Get a copy of plate_image
    copy_plate = plate_image.copy()

    # Get shape of plate_image
    Height, Width = plate_image.shape[:2]

    # Mask need 2 extra pixels
    mask = np.zeros((Height+2, Width+2), np.uint8)

    for i, rect in enumerate(rects):
        # Take advantage of white background
        # Use floodFill to retrieve more precise contour box
        
        w, h = (int(x) for x in rect[1:3])

        # Draw center point of rotated Rect
        cx, cy = int(rect[0] + w/2), int(rect[1] + h/2)
        cv2.circle(plate_image, (cx, cy), 3, (0, 255, 0), -1) # Color and -1 mean filled circle
        
        # Get min size between width and height
        minSize = w if w < h else h
        minSize = minSize - minSize*0.5
        
        # Init floodFill parameters
        # Mask need 2 extra pixels
        mask = np.zeros((Height+2, Width+2), np.uint8)
        seed_pt = None
        connectivity = 4 # Fours neighbours floodFill
        loDiff = 30; upDiff = 30
        newMaskVal = 255
        numSeeds = 250

        # Create flags
        # cv2.FLOODFILL_FIXED_RANGE: if set, the difference between the current pixel and seed pixel is considered (neighbor pixels too)
        # cv2.FLOODFILL_MASK_ONLY: if set, function dont change the image (newVal is ignored)
        flags = connectivity + (newMaskVal << 8) + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY
        # Generate several seeds near center of rotated rect
        for j in range(numSeeds):
            minX = rect[0][0] - minSize/2 - 25 if rect[0][0] - minSize/2 - 25 >= 0 else rect[0][0] - minSize/2
            maxX = rect[0][0] + (minSize/2) + 25 if rect[0][0] + (minSize/2) + 25 < Width else rect[0][0] + minSize/2
            minY = rect[0][1] - minSize/2 - 10 if rect[0][1] - minSize/2 - 10 >= 0 else rect[0][1] - minSize/2
            maxY = rect[0][1] + (minSize/2) + 10 if rect[0][1] + (minSize/2) + 10 < Height else rect[0][1] + (minSize/2)
            
            # print("Size ", w, h)

            # minX = minSize/2
            # maxX = w - minSize/4
            # minY = minSize/2
            # maxY = h - minSize/2

            # Use minSize to generate random seed near center of rect
            seed_ptX = np.random.randint(minX, maxX)
            seed_ptY = np.random.randint(minY, maxY)
            
            seed_pt = (seed_ptX, seed_ptY)
            
            # If seed value is not 0, pass it
            if seed_ptX >= Width or seed_ptY >= Height or copy_plate[seed_ptY, seed_ptX] != 0:
                continue

            # Draw seeds and floodFill
            cv2.circle(plate_image, seed_pt, 1, (0,255,255), -1)
            # cv2.floodFill(plate_image, mask, seed_pt, (255, 0, 0), loDiff, upDiff, flags)
            cv2.floodFill(plate_image, mask, seed_pt, 255, loDiff, upDiff, flags)
            
        
        cv2.imshow("Mask", mask)
        plate_image = np.copy(copy_plate)
    
    # # Get all point represent foreground
    # pointInterrest = np.where(mask == 255)
    # # Turn x's array and y's array index into xy pair index and format it
    # pointInterrest = np.array(list(zip(pointInterrest[0], pointInterrest[1]))).reshape(-1, 1, 2)
    
    # # Create rotated rect
    # minRect = cv2.minAreaRect(pointInterrest)
    
    # # # Get rotation matrix of rect
    # # r = minRect[1][0] / minRect[1][1]
    # # angle = minRect[2]
    # # print(angle)
    # # if r < 1:
    # #     angle = 90 + angle
    
    # # rotmat = cv2.getRotationMatrix2D(minRect[0], angle, 1)

    # # # Rotate image
    # # rotated_img = cv2.warpAffine(plate_image, rotmat, plate_image.shape, cv2.INTER_CUBIC)

    # # Crop image
    # crect_x, crect_y = (int(x) for x in minRect[1])
    # crop_img = cv2.getRectSubPix(rotated_img, (crect_x, crect_y), minRect[0])

    # cv2.imshow("Crop", crop_img)
    return mask

   

# Load cascade model
cascade = cv2.CascadeClassifier(r"E:\K16\Junior\TGMT\ALPR-project\Version 2\haarcascade_russian_plate_number.xml") # haarcascade_licence_plate_rus_16stages.xml

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
img = cv2.imread(r"E:\K16\Junior\TGMT\ALPR-project\test_images\DSCN0415.jpg")

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