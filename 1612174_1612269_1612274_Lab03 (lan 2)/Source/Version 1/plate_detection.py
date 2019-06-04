import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


class PlateDetection:

    def __init__(self, plateHeight = 11, plateWidth = 52):
        # Error margin: 40%
        self.error_margin = 0.4
        # Aspect ratio of plate depend the plate size on each country
        # Viet Nam plate size: 14x19 => aspect ratio = 19/14 = 1.36
        self.plate_asp_ratio = float(plateWidth) / plateHeight


    # Preliminary validation before classification: Verify size bonding rect base on area and aspect ratio
    def verify_sizes(self, candidate_rect):
        # Range of area: min = 15, max = 125 pixels
        min_area = 15*self.plate_asp_ratio*15
        max_area = 125*self.plate_asp_ratio*125
        # accept patches such that aspect ratio of bouding rect is in range of min_ratio and max_ratio
        min_ratio = self.plate_asp_ratio - self.plate_asp_ratio*self.error_margin
        max_ratio = self.plate_asp_ratio + self.plate_asp_ratio*self.error_margin

        # Get canWidth and canHeight
        canWidth, canHeight = candidate_rect[1]
        if canHeight == 0 or canWidth == 0:
            return False

        # Check angle for rotated rect
        if len(candidate_rect) == 3:
            # Note angle is counter-clockwise and measured along the longer side
            angle = candidate_rect[2]
            if canWidth >= canHeight:
                angle *= -1
            else: # <
                angle = 90 - angle
            
            # Angle is larger than 30 degree (Range: 0 - 180 degree)
            if 30 < abs(angle) < 150:
                return False

        # Compute area of this bounding rect
        candidate_area = canWidth * canHeight
        # Compute aspect ratio of bounding rect
        candidate_ratio = canWidth / canHeight

        # If width < height then reverse it
        if candidate_ratio < 1:
            candidate_ratio = 1/candidate_ratio

        '''
        If candidatte_area is out ot range considered area 
        or candidate_ratio is out of range of considered ratio range
        '''
        if candidate_area < min_area or candidate_area > max_area \
            or candidate_ratio < min_ratio or candidate_ratio > max_ratio:
            return False
        return True


   # Use floodFill algorithm for more precise cropping
    def floodFill(self, rects, plate_image, step_by_step):
        # Output
        eqhist_img_list = []
        plate_img_list = []

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
            # maximal lower and maximal upper brightness/color difference between the pixel to fill 
            # and the pixel neighbors or seed pixel.
            loDiff = 30; upDiff = 30
            newMaskVal = 255 # New color to fill
            numSeeds = 200

            # Create flags
            # cv2.FLOODFILL_FIXED_RANGE: if set, the difference between the current pixel and seed pixel is considered (neighbor pixels too)
            # cv2.FLOODFILL_MASK_ONLY: if set, function dont change the image (newVal is ignored)
            flags = connectivity + (newMaskVal << 8) + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY
            # Generate several seeds near center of rotated rect
            for j in range(numSeeds):
                minX = rect[0][0] - minSize + minSize/2 - 30
                maxX = rect[0][0] + minSize - (minSize/2) + 50
                minY = rect[0][1] - minSize + minSize/2
                maxY = rect[0][1] + minSize - (minSize/2)
                
                # Use minSize to generate random seed near center of rect
                seed_ptX = np.random.randint(minX, maxX)
                seed_ptY = np.random.randint(minY, maxY)
                
                seed_pt = (seed_ptX, seed_ptY)
                
                # # If seed value is not 0, pass it
                # if seed_ptX >= Width or seed_ptY >= Height or copy_plate[seed_ptY, seed_ptX] != 0:
                #     continue

                # Draw seeds and floodFill
                cv2.circle(plate_image, seed_pt, 1, (0,255,255), -1)
                cv2.floodFill(copy_plate, mask, seed_pt, (255, 0, 0), (loDiff,)*3, (upDiff,)*3, flags)
                
            
            if step_by_step:
                cv2.imshow("Mask " + str(i), mask)
            
            # Get all point represent foreground (Point of interest)
            poi = np.where(mask == 255)
            # Turn x's array and y's array index into xy pair index and format it
            poi = np.array(list(zip(poi[1], poi[0]))).reshape(-1, 1, 2)
            
            # Create rotated rect
            minRect = cv2.minAreaRect(poi)

            if self.verify_sizes(minRect):
                # Get bounding box of minRect
                box = cv2.boxPoints(minRect)
                box = np.int0(box)
                cv2.drawContours(plate_image, [box], 0, (0, 0, 255), 2)

                if step_by_step:
                    cv2.imshow("Minimal area rectangle " + str(i), plate_image)

                # Get rotation matrix of rect
                r = minRect[1][0] / minRect[1][1]
                angle = minRect[2]
                
                # Note that: Angle is counter-clockwise and measured along the longer edge
                if r < 1:
                    angle = 90 + angle
                
                '''
                Calculates an affine matrix of 2D rotation
                --------
                Input:
                    Center
                    Angle: counter-clockwise rotation
                    Scale
                Output:
                    mapMatrix: The output affine transformation, 2x3 floating-point matrix
                    [alpha   beta    (1-alpha)(center.x) - beta*center.y
                    -beta   alpha   beta*center.x + (1-alpha)*center.y]

                    where
                        alpha = scale*cos(angle)
                        beta = scale*sin(angle)
                For more detail:
                https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpaffine
                '''
                rotmat = cv2.getRotationMatrix2D(minRect[0], angle, 1)

                '''
                Applies an affine transformation to an image.
                --------
                Input:
                    src: input image
                    M: the 2x3 affine transformation matrix
                    dsize: size of output image
                    flags: interpolation method we use (resize image)
                    borderMode and borderValue if needed
                Output:
                    dst: output image

                For more detail:
                https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpaffine
                '''
                rotated_img = cv2.warpAffine(copy_plate, rotmat, copy_plate.shape[:2], cv2.INTER_CUBIC)

                # Crop image 
                rect_w, rect_h = 0, 0
                if r < 1:
                    # If width < height, we need to invert it
                    rect_h, rect_w = (int(x) for x in minRect[1])
                else:
                    rect_w, rect_h = (int(x) for x in minRect[1])
                '''
                Retrives a pixel rectangle from an image with sub-pixel accuracy
                Input:
                    src: Input image
                    patchSize: Size of extracted patch
                    center: floating point coordinates
                Output:
                    dst: Extracted patch

                For more details:
                https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpaffine
                '''
                crop_img = cv2.getRectSubPix(rotated_img, (rect_w, rect_h), minRect[0])
                if step_by_step:
                    cv2.imshow("Crop " + str(i), crop_img)

                '''
                Because plate image in each case may have different conditions: light, scale
                First, we need to resize plate image to have the same size and use equalize histogram to balance the brightness and increase the contrast
                '''
                resized_img = cv2.resize(crop_img, (144, 33), 0, 0, interpolation = cv2.INTER_CUBIC)
                # Second, Equalizes histogram
                # Gray-scale
                gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                # Gaussian blur
                blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0) # sigma = 0: function will choose sigma base on kernel size
                eqhist_img = cv2.equalizeHist(blur_img)
                
                if step_by_step:
                    cv2.imshow("EQ plate " + str(i), eqhist_img)

                eqhist_img_list.append(eqhist_img)
                plate_img_list.append(resized_img)
                
        return eqhist_img_list, plate_img_list

    def process_plate_image(self, plateImg):
        # Convert to Gray-Scale
        grayPlate = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)
        # Blur image
        # grayPlate = cv2.GaussianBlur(grayPlate, (5, 5), 1.0) # remove noise
        # Dilate plate image for OCR
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
        dilateImg = cv2.dilate(grayPlate, kernel, iterations = 1)
        # Threshold image
        _, threshImg = cv2.threshold(dilateImg, 150, 255, cv2.THRESH_BINARY)
        #ret, threshImg = cv2.threshold(dilateImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Find contours
        contours, hierachy = cv2.findContours(np.copy(threshImg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get corner points on boundary
        # tmpImg = plateImg.copy()
        # cv2.drawContours(tmpImg,contours,-1,(0,255,0),3)
        # cv2.imshow('Contours',tmpImg)

        # If contours exists
        if len(contours) > 0:
            # Compute Area of each contour (Moment M[00])
            area_list = [cv2.contourArea(cnt) for cnt in contours]

            # Find index of contour has largest area
            imax = np.argmax(area_list)
            
            # Get bouding rect of this contour
            x, y, w, h = cv2.boundingRect(contours[imax])
            # Verify plate image again
            if not(self.verify_sizes([(x, y), (w, h)])):
                return None, None
            
            # Crop region that have number in image
            found_plate = np.copy(plateImg[y:y+h, x:x+w])
            #cv2.imshow('Found plate',found_plate)
            return found_plate, [x, y, w, h]
        return None, None

    

    # Detect plate in image
    '''
    Input: 
        image: BGR image
        step_by_step: Show step by step works (default = False)
    Output:
        list of possible plates
    '''
    def detect_plates(self, img, step_by_step = False): 
        listOfPlates = []
        

        # Resize image: width = 600, height = (original_height * 600 / orignal_width
        # INTER_AREA mode: Resampling using pixel area relation
        height, width = img.shape[:2]
        img = cv2.resize(img, (600, round(height*600/width)), interpolation = cv2.INTER_AREA)

        #Create copy of original image
        copy_org_img = img.copy()

        # Convert to grayscale
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if step_by_step:
            plt.figure()
            plt.imshow(grayImg, cmap = 'gray', interpolation="bicubic")
            plt.title("Gray-scale image"); plt.axis("off")

        # Step 1: Blur image
        blurImg = cv2.GaussianBlur(grayImg, (5, 5), 1.0) # remove noise

        if step_by_step:
            plt.figure()
            plt.imshow(blurImg, cmap = 'gray', interpolation="bicubic")
            plt.title("Gaussian-blur image"); plt.axis("off")

        # Step 2: Find vertical edges
        sobelX = cv2.Sobel(blurImg, cv2.CV_8UC1, dx=1, dy=0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)

        if step_by_step:
            plt.figure()
            plt.imshow(sobelX, cmap = 'gray', interpolation="bicubic")
            plt.title("Vertical sobel"); plt.axis("off")

        # Step 3: Threshold sobelX with Otsu algorithm (Otsu will automaticly find optimal threshold value)
        ret, thresholdImg = cv2.threshold(sobelX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if step_by_step:
            plt.figure()
            plt.imshow(thresholdImg, cmap='gray', interpolation="bicubic")
            plt.title("Threshold image by Otsu algorithm")
            plt.axis("off")

        # Step 4: Apply morphological operation to remove blank spaces between each vertical sobel
        # First It do dilation followed by Erosion then to remove small holes inside foreground object
        # Called Closing method.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3))
        morphImg = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, kernel = rect_kernel)

        if step_by_step:
            plt.figure()
            plt.imshow(morphImg, cmap='gray', interpolation="bicubic")
            plt.title("Morphological image")
            plt.axis("off")

        # When we do morphological operation, we now have regions that maybe contain plate.
        # But most of them don't contain plate, we need to refine them.
        
        # Step 5: Find contours of possible plates
        # Just need external contours
        """Output: 
            (this function will modified directly on source img)
            - Contours contain all the boundary points of contours
            - Hierarchy contain information about the image topology"""
        # Copy morphImg
        morphImg_copy = np.copy(morphImg)
        
        contours, hierarchy = cv2.findContours(morphImg_copy,             # Source Image
                                            cv2.RETR_EXTERNAL,  # Extract external contours
                                            cv2.CHAIN_APPROX_SIMPLE)  # get corner points of each contour
        
        if step_by_step:
            # Draw all contours
            """Input:
                source image: draw contour inside this img
                contours: list of contour points
                -1: mean draw all contours (or individual contour)
                color and thickness
            """
            copy_img = np.copy(img)
            cv2.drawContours(copy_img, contours, -1, (0, 255, 0), 2)

            plt.figure()
            plt.imshow(cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB))
            plt.title("Contours")
            plt.axis("off")

        

        # Extract and refine minimal rect area (rotated object)
        rect_list = self.extract_and_refine_bounding_rect(contours)
        
        if step_by_step:
            copy_img = np.copy(img)
            # Draw rects
            for rect in rect_list:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(copy_img, [box], 0, (0, 255, 0), 2)

            plt.figure()
            plt.imshow(cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB))
            plt.title("Rotated rectangles (minimal rectangles)")
            plt.axis("off")

        output = self.floodFill(rect_list, copy_org_img, step_by_step)
        
        # If existing candidate
        if len(output[0]) > 0:
            return output[0], output[1]
            
        # # Sort contour base on area (Descending order)
        # contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        # for (i,cnt) in enumerate(contours):
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     plateImg = np.copy(img[y:y+h, x:x+w])

        #     if step_by_step:
        #         plt.figure()
        #         plt.imshow(cv2.cvtColor(plateImg, cv2.COLOR_BGR2RGB))
        #         plt.title("Possible plates")
        #         plt.axis("off")

        #     cv2.imshow('Detected Plate ' + str(i), plateImg)
        #     # Process image with some stuffs: dilate, threshold, contours and verify size to refine plate image
        #     found_plate, plat_rect = self.process_plate_image(plateImg)
        #     if found_plate is None:
        #         continue

        #     listOfPlates.append(found_plate)

        # return listOfPlates

        return [], []

        

    # For each contour, extract the bouding rectangle of minimal area
    # And also do small validatation before classfying them 
    def extract_and_refine_bounding_rect(self, contours):
        i = 0
        rect_list = [] # List to store refined rect
        while i < len(contours):
            # Create bounding rect of object
            # Bouding rect contain: [center (x, y), width&height (w,h), angle)
            bounding_rect = cv2.minAreaRect(contours[i])
            # Validate bounding_rect base on its area and aspect ratio
            if(not(self.verify_sizes(bounding_rect))):
                """??? VARIABLE BY REFERENCE ???"""
                contours.pop(i)
                continue
            
            # Push this bounding rect into rect list
            rect_list.append(bounding_rect)
            # Increase i
            i = i + 1
       
        return rect_list


    def makeBoundingBoxOfCandidateContours(self, candidate_cnts):
        boundingBox = []
        for cnt in candidate_cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            boundingBox.append([(x, y), (y+h, x+w)])
        
        return boundingBox