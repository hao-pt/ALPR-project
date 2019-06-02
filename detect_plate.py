import cv2
import numpy as np
from PIL import Image, ImageFilter
from pytesseract import image_to_string


# Use floodFill algorithm for more precise cropping
def floodFill(rects, plate_image):
    # Get a copy of plate_image
    copy_plate = plate_image.copy()

    # Get shape of plate_image
    Height, Width = plate_image.shape[:2]

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
        numSeeds = 200

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

def preprocess_image(found_plate, plate_box):
    # Resize image: Scale down a half
    found_plate = cv2.resize(found_plate, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
    cv2.imshow("Scale down a half", found_plate)

    copy_plate = found_plate.copy()

    # Convert to gray image
    grayPlate = cv2.cvtColor(found_plate, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray plate", grayPlate)

    # Equalize brightness and increase the contrast
    grayPlate = cv2.equalizeHist(grayPlate)
    cv2.imshow("Equalize histogram", grayPlate)

    # Blur image
    blurPlate = cv2.GaussianBlur(grayPlate, (5, 5), 0) # Remove noise with sigma based on kernel size
    cv2.imshow("Gaussian-blur image", blurPlate)    

    # Apply closing operation:  Dilation followed by Erosion to remove noise and hole inside object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closingImg = cv2.morphologyEx(blurPlate, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing image", closingImg)

    # Apply threshold to get image with only b&w (binarization)
    # _, threshPlate = cv2.threshold(grayPlate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshPlate = cv2.adaptiveThreshold(closingImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 22)
    cv2.imshow("Threshold", threshPlate)

    # Find contours of each character and extract it
    contours, _ = cv2.findContours(threshPlate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get only corners of each contour

    # Get rotated rectangle of each contours
    rects = []
    for i in range(len(contours)):
        # Bouding rect contain: [center (x, y), width&height (w,h), angle)
        rect = cv2.minAreaRect(contours[i])
        
        if rect[1][0] == 0 or rect[1][1] == 0 or rect[1][0] < 100 or rect[1][1] < 80:
            continue

        rects.append(rect)
    
    # # Sort contours base on boundingRect position x
    # contours = sorted(contours, key = lambda cnt: cv2.boundingRect(cnt)[0])
    
    # Draw contour
    cv2.drawContours(found_plate, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", found_plate)

    mask = floodFill(rects, threshPlate)
    mask = cv2.bitwise_not(mask)
    
    cv2.imshow("Mask inverse", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # get only corners of each contour
    cv2.drawContours(copy_plate, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Mask contour", copy_plate)

    return contours, mask, threshPlate

def character_separation(mask, contours, threshPlate):
    character_img = []
    for i, cnt in enumerate(contours):
        # Get bounding box
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        if not(300 <= w*h <= 1250 and w <= h):
            continue

        # character roi
        character_roi = np.copy(threshPlate[y:y+h, x:x+w])
        character_roi = cv2.resize(character_roi, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
        # Make border: 10 extra with constant value = 255
        character_roi = cv2.copyMakeBorder(character_roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = 255)

        # Threshold it
        charImg = cv2.GaussianBlur(character_roi, (3,3), 0)
        charImg = cv2.adaptiveThreshold(charImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)
        
        # Push to list
        character_img.append(charImg)

        cv2.imshow("{:d}".format(i), charImg)

        # Draw bounding box of each character
        rect = cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('rect', rect)

    return character_img
    


def character_recognition(found_plate, plate_box, img):
    # Preprocess plate image to find contours of each character in plate
    contours, mask, threshPlate = preprocess_image(found_plate, plate_box)
    # Separate character in plate
    character_img = character_separation(mask, contours, threshPlate)

    plate_text = ""
    for charImg in character_img:
        # Store image in PIL image format
        pilImg = Image.fromarray(charImg)

        # Recognize text with Tesseract
        c = image_to_string(pilImg, lang = "eng")
        c = plate_text.replace(" ", "") # Remove space

        if len(c) == 0:
            c = "?"

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
cascade = cv2.CascadeClassifier(r"E:\K16\Junior\TGMT\ALPR-project\GreenParking_num-3000-LBP_mode-ALL_w-30_h-20.xml")

# Load image
img = cv2.imread(r"E:\K16\Junior\TGMT\ALPR-project\Bike_back\17.jpg")

# Resize image
width = 600
height = int(600 * img.shape[0] / img.shape[1]) 
img = cv2.resize(img, (width, height))

# Convert to gray image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Equalize brightness and increase the contrast
grayImg = cv2.equalizeHist(grayImg)

# detect license plate
box_plates = cascade.detectMultiScale(grayImg, scaleFactor = 1.1, minNeighbors = 3, flags = cv2.CASCADE_SCALE_IMAGE)

# Draw bounding box on detected plate
for (x, y, w, h) in box_plates:
    # Draw bounding box
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Plate roi
    plate_roi = np.copy(img[y:y+h, x:x + w])
    cv2.imshow("Plate ROI", plate_roi)
    # OCR
    character_recognition(plate_roi, (x, y, w, h), img)

# cv2.imshow("Plate detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()