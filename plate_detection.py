# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from pytesseract import image_to_string

'''
License plate number recognition by Team Fusion
Ref:
    [1] Baggio, D. L. (2012). 5. Number Plate Recognition Using SVM and Neural Networks. 
    In Mastering OpenCV with Practical Computer Vision Projects (6th ed., pp. 161-188). 
    Birmingham, UK: Packt Publishing
    [2] https://github.com/kagan94/Automatic-Plate-Number-Recognition-APNR.git
'''

class CPlateDetection:
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
                contours.pop(i)
                continue
            
            # Push this bounding rect into rect list
            rect_list.append(bounding_rect)
            # Increase i
            i = i + 1
       
        return rect_list

    # Phan nay dung floodFill nhung no phuc tap qua, tui chua nam ro (Tam thoi bo qua)
    """
    # Extract license plate region in image
    def plate_extraction(self, img, rect_list):
        for i, rect in enumerate(rect_list):
            # Take advantage of white background property
            # To crop rect with better accuracy
            # Use floodfill algorithm to retrieve the rotated rectagle
            # First step, put several seeds (circles) near the center of rotated rect
            cv2.circle(img, rect[0], 3, (0, 255, 0), -1)
            # Get min size between width and height
            minSize = (rect[1][0]) if rect[1][0] < rect[1][1] else rect[1][1]
            minSize = minSize - minSize*0.5
    """

    # Process plate image with some stuffs: dilate, threshold, contours and verify size to refine plate image
    def process_plate_image(self, plateImg):
        # Convert to Gray-Scale
        grayPlate = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)
        # Blur image
        grayPlate = cv2.GaussianBlur(grayPlate, (5, 5), 1.0) # remove noise
        # Dilate plate image for OCR
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
        dilateImg = cv2.dilate(grayPlate, kernel, iterations = 1)
        # Threshold image
        _, threshImg = cv2.threshold(dilateImg, 150, 255, cv2.THRESH_BINARY)
        #ret, threshImg = cv2.threshold(dilateImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Find contours
        contours, hierachy = cv2.findContours(np.copy(threshImg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get all points on boundary
        
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
            return found_plate, [x, y, w, h]
        
        return None, None


    # Detect plate in image
    def plate_detection(self, img): 
        # Resize image: width = 400, height = (original_height * 400 / orignal_width
        # INTER_AREA mode: Resampling using pixel area relation
        height, width = img.shape[:2]
        img = cv2.resize(img, (400, round(height*400/width)), interpolation = cv2.INTER_AREA)

        # Convert to grayscale
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Blur image
        blurImg = cv2.GaussianBlur(grayImg, (5, 5), 1.0) # remove noise
        # Step 2: Find vertical edges
        sobelX = cv2.Sobel(blurImg, cv2.CV_8UC1, dx=1, dy=0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)

        # plt.figure()
        # plt.imshow(sobelX, cmap = 'gray', interpolation="bicubic")
        # plt.title("Vertical sobel"); plt.axis("off")

        # Step 3: Threshold sobelX with Otsu algorithm (Otsu will automaticly find optimal threshold value)
        ret, thresholdImg = cv2.threshold(
            sobelX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        plt.figure()
        plt.imshow(thresholdImg, cmap='gray', interpolation="bicubic")
        plt.title("Threshold image by Otsu algorithm")
        plt.axis("off")

        # Step 4: Apply morphological operation to remove blank spaces between each vertical sobel
        # First It do dilation followed by Erosion then to remove small holes inside foreground object
        # Called Closing method.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morphImg = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, kernel = rect_kernel)

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
                                            cv2.CHAIN_APPROX_NONE)  # get all points of each contour
        
        # Draw all contours
        """Input:
            source image: draw contour inside this img
            contours: list of contour points
            -1: mean draw all contours (or individual contour)
            color and thickness
        """
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # plt.figure()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title("Contours image")
        # plt.axis("off")

        

        # Extract and refine minimal rect area (rotated object)
        rect_list = self.extract_and_refine_bounding_rect(contours)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            plateImg = np.copy(img[y:y+h, x:x+w])

            # plt.figure()
            # plt.imshow(cv2.cvtColor(plateImg, cv2.COLOR_BGR2RGB))
            # plt.title("Plates")
            # plt.axis("off")

            # Process image with some stuffs: dilate, threshold, contours and verify size to refine plate image
            found_plate, plat_rect = self.process_plate_image(plateImg)
            if found_plate is None:
                continue
            
            # Take advantage of white background, so we will check if the average color >= 100, it will a plate
            if np.mean(found_plate).astype(np.uint8) >= 100:
                # Display plate
                plt.figure()
                plt.imshow(cv2.cvtColor(found_plate, cv2.COLOR_BGR2RGB))
                plt.title("Plates")
                plt.axis("off")

                # ----------------------Now we use Tesseract to recognize charactor in plate-----------------------
                # Convert to gray image
                grayPlate = cv2.cvtColor(found_plate, cv2.COLOR_BGR2GRAY)
                # Blur image
                blurPlate = cv2.GaussianBlur(grayImg, (5, 5), 1.0) # Remove noise
                # Apply closing operation:  Dilation followed by Erosion to remove noise and hole inside object
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                closingImg = cv2.morphologyEx(blurPlate, cv2.MORPH_CLOSE, kernel)
                # Apply threshold to get image with only b&w (binarization)
                threshPlate = cv2.threshold(closingImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Store image in PIL image format
                # pilImg = Image.fromarray(found_plate)

                # # Recognize text with Tesseract
                # plate_text = image_to_string(found_plate, lang = "eng")
                # plate_text = plate_text.replace(" ", "") # Remove space
                # print(plate_text)


        # Draw rects
        # for rect in rect_list:
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        # plt.figure()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title("Rotated rectangles")
        # plt.axis("off")

# Measure time
e1 = cv2.getTickCount()        

# Read image
img = cv2.imread(".\\test_images\\9773BNB.jpg")  # r".\Bike_back\1.jpg"

# Declare CPlateDetection obj
plateDetector = CPlateDetection()
plateDetector.plate_detection(img)


# # Display image
# cv2.namedWindow("Blur", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("Blur", morphImg)

# # Wait key
# cv2.waitKey()

# End time
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print('Time: %.2f(s)' %(time))

plt.show()
