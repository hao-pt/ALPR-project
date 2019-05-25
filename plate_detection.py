# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt



class CPlateDetection:
    def __init__(self, plateHeight = 14, plateWidth = 19):
        # Error margin: 40%
        self.error_margin = 0.4
        # Aspect ratio of plate depend the plate size on each country
        # Viet Nam plate size: 14x19 => aspect ratio = 19/14 = 1.36
        self.plate_asp_ratio = plateWidth/plateHeight

    # Preprocess: Verify size bonding rect base on area and aspect ratio
    def verify_sizes(self, candidate_rect):
        # Range of area: min = 15, max = 125 pixels
        min_area = 15*self.plate_asp_ratio*15
        max_area = 125*self.plate_asp_ratio*125
        # accept patches such that aspect ratio of bouding rect is in range of min_ratio and max_ratio
        min_ratio = self.plate_asp_ratio - self.plate_asp_ratio*self.error_margin
        max_ratio = self.plate_asp_ratio + self.plate_asp_ratio*self.error_margin

        # Get canWidth and canHeight
        canWidth, canHeight = candidate_rect[1][:]
        if canHeight == 0:
            return False

        # Compute area of this bounding rect
        candidate_area = canWidth * canHeight
        # Compute aspect ratio of bounding rect
        candidate_ratio = canWidth / canHeight

        # If width < height then reverse it
        if candidate_ratio < 1:
            candidate_ratio = 1/candidate_ratio

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
            rect_list.append(contours[i])
            # Increase i
            i = i + 1
        print(len(rect_list))
        return rect_list

    # Extract license plate region in image
    def plate_extraction(self, img, rect_list):
        for i, rect in enumerate(rect_list):
            cv2.circle(result, rect)



    # Detect plate in image
    def plate_detection(self, img):
        # Convert to grayscale
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur image
        blurImg = cv2.GaussianBlur(grayImg, ksize=(5, 5), sigmaX=1.0)
        # Find vertical edges
        sobelX = cv2.Sobel(blurImg, cv2.CV_8U, dx=1, dy=0)

        # plt.figure()
        # plt.imshow(sobelX, cmap = 'gray', interpolation="bicubic")
        # plt.title("Vertical sobel"); plt.axis("off")

        # Threshold sobelX with Otsu algorithm (Otsu will automaticly find optimal threshold value)
        ret, thresholdImg = cv2.threshold(
            sobelX, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        plt.figure()
        plt.imshow(thresholdImg, cmap='gray', interpolation="bicubic")
        plt.title("Threshold image by Otsu algorithm")
        plt.axis("off")

        # Apply morphological operation to remove blank spaces between each vertical sobel
        # First It do dilation followed by Erosion then to remove small holes inside foreground object
        # Called Closing method.
        morphImg = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, kernel=(17, 3))

        plt.figure()
        plt.imshow(morphImg, cmap='gray', interpolation="bicubic")
        plt.title("Morphological image")
        plt.axis("off")

        # When we do morphological operation, we now have regions that maybe contain plate.
        # But most of them don't contain plate, we need to refine them.
        # Find contours of possible plates
        # Just need external contours
        """Output: 
            (this function will modified directly on source img)
            - Contours contain all the boundary points of contours
            - Hierarchy contain information about the image topology"""
        contours, hierarchy = cv2.findContours(morphImg,             # Source Image
                                            cv2.RETR_EXTERNAL,  # Extract external contours
                                            cv2.CHAIN_APPROX_NONE)  # get all points of each contour
        
        # Draw all contours
        """Input:
            source image: draw contour inside this img
            contours: list of contour points
            -1: mean draw all contours (or individual contour)
            color and thickness
        """
        # #for contour in contours:
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # plt.figure()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title("Contours image")
        # plt.axis("off")

        # Extract and refine minimal rect area (rotated object)
        rect_list = self.extract_and_refine_bounding_rect(contours)
        
        

# Read image
img = cv2.imread(r".\Bike_back\1.jpg")

# Declare CPlateDetection obj
plateDetector = CPlateDetection(1, 1)
plateDetector.plate_detection(img)


    


# # Display image
# cv2.namedWindow("Blur", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("Blur", morphImg)

# # Wait key
# cv2.waitKey()

plt.show()
