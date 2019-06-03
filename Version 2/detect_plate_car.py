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
        
        # Draw center point of rotated Rect
        cx, cy = (int(x) for x in rect[0])
        cv2.circle(plate_image, (cx, cy), 3, (0, 255, 0), -1) # Color and -1 mean filled circle

        w, h = (int(x) for x in rect[1])
        
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
            minX = rect[0][0] - minSize/2 - 40
            maxX = rect[0][0] + (minSize/2) + 50
            minY = rect[0][1] - minSize/2 - 10
            maxY = rect[0][1] + (minSize/2) + 10
            
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
        # plate_image = np.copy(copy_plate)
    
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


def preprocess_image(found_plate, plate_box):
    # # Resize image: Scale down a half
    # found_plate = cv2.resize(found_plate, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
    # cv2.imshow("Scale down a half", found_plate)

    copy_plate = found_plate.copy()

    # Convert to gray image
    grayPlate = cv2.cvtColor(found_plate, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray plate", grayPlate)

    # Equalize brightness and increase the contrast
    grayPlate = cv2.equalizeHist(grayPlate)
    cv2.imshow("Equalize histogram", grayPlate)

    # Blur image
    blurPlate = cv2.GaussianBlur(grayPlate, (3, 3), 0) # Remove noise with sigma based on kernel size
    cv2.imshow("Gaussian-blur image", blurPlate)    

    # Apply closing operation:  Dilation followed by Erosion to remove noise and hole inside object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closingImg = cv2.morphologyEx(blurPlate, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing image", closingImg)

    # Apply threshold to get image with only b&w (binarization)
    # _, threshPlate = cv2.threshold(grayPlate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshPlate = cv2.adaptiveThreshold(closingImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    cv2.imshow("Threshold", threshPlate)

    copy_thresh = threshPlate.copy()

    # Find contours of each character and extract it
    contours, _ = cv2.findContours(threshPlate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get only corners of each contour

    # Get rotated rectangle of each contours
    rects = []
    for i in range(len(contours)):
        # Bouding rect contain: [center (x, y), width&height (w,h), angle)
        rect = cv2.minAreaRect(contours[i])
        if rect[1][0] == 0 or rect[1][1] == 0 or rect[1][0] < 80 or rect[1][1] < 15:
            continue

        rects.append(rect)
    
    # Draw contour
    cv2.drawContours(found_plate, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", found_plate)

    # FloodFill to extract character
    mask = floodFill(rects, threshPlate)
    mask = cv2.bitwise_not(mask) # invert it
    
    cv2.imshow("Mask inverse", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # get only corners of each contour
    
    # Filter contour (just get contour of character)
    refine_contours = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # print(w, h)
        if 150 <= w*h <= 500 and w <= h:
            refine_contours.append(cnt)
            
    cv2.drawContours(copy_plate, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Mask contour", copy_plate)

    return refine_contours, mask, copy_thresh

def character_separation(mask, contours, plate_img):
    character_img = []
    for i, cnt in enumerate(contours):
        # Get bounding box
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # character roi
        character_roi = np.copy(plate_img[y:y+h, x:x+w])

        # gray-scale image
        character_roi = cv2.cvtColor(character_roi, cv2.COLOR_BGR2GRAY)
        character_roi = cv2.resize(character_roi, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_CUBIC)
        # Make border: 10 extra with constant value = 255
        character_roi = cv2.copyMakeBorder(character_roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = 255)

        # cv2.imshow("Character {:d}".format(i), character_roi)
        # character_img.append(character_roi)

        # Apply dilation operation:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        openingImg = cv2.morphologyEx(character_roi, cv2.MORPH_OPEN, kernel)

        # Threshold it
        charImg = cv2.GaussianBlur(openingImg, (3,3), 0)
        charImg = cv2.adaptiveThreshold(charImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

        cv2.imshow("Character {:d}".format(i), charImg)
        character_img.append(charImg)

        # Draw bounding box of each character
        rect = cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        cv2.imshow('rect', rect)

    return character_img
    
def character_recognition(found_plate, plate_box, img):
    copy_plate = found_plate.copy()

    # Preprocess plate image to find contours of each character in plate
    contours, mask, plate_img = preprocess_image(found_plate, plate_box)
    
    # Sort contours base on boundingRect position x
    contours = sorted(contours, key = lambda cnt: cv2.boundingRect(cnt)[0])
    # cv2.imshow("PPPP", found_plate)
    # Separate character in plate
    character_img = character_separation(mask, contours, found_plate)

    plate_text = ""
    config = ("-l eng --oem 1 --psm 3")
    for i, charImg in enumerate(character_img):
        # Store image in PIL image format
        # pilImg = Image.fromarray(charImg)
        
        # Recognize text with Tesseract
        c = image_to_string(charImg, config = config)
        c = plate_text.replace(" ", "") # Remove space

        if len(c) == 0:
            c = "?"
        if len(c) > 0:
            c = c[0]

        plate_text += c

    # Get bounding box position of plate
    x, y, w, h = plate_box

    # Print Recognize text on image
    copy_img = np.copy(img)
    # Specify font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # draw text
    cv2.putText(copy_img, plate_text, (x, y + h + 50), font, 1.0, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("License plate number recognition", copy_img)
    
    print(plate_text)   

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
img = cv2.imread(r"E:\K16\Junior\TGMT\ALPR-project\test_images\5445BSX.JPG")

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
    character_recognition(plate_roi, (x, y, w, h), img)

# classify_plate()

cv2.imshow("Plate detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()