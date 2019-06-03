import numpy as np
import cv2


class CharactersSeparation:

    def __init__(self):
        self.init = 0    
    
    # checking the posibility of being character based on image's area
    def verify_size_character(self, img):
        #grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshImg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
        aspect = float(45/77)
        height,width = threshImg.shape
        charAspect = float(width/height)
        error = 0.35
        minHeight, maxHeight = 30, 56
        minAspect, maxAspect = 0.2, aspect + aspect*error

        area = cv2.countNonZero(threshImg)
        bbArea = width * height

        percPixels = float(area/bbArea)
        if percPixels < 0.8 and charAspect > minAspect and charAspect < maxAspect \
            and height >= minHeight and height < maxHeight:
            return True
        return False

    
    # Separating each character in plate image
    def get_character_images(self, plateImg):
    #def process_plate_image(self,plateImg):
        ret = [] # list of character images
        # Convert to Gray-Scale
        height,width = plateImg.shape[:2]
        plateImg = cv2.resize(plateImg,(2*width,2*height))
        plateImg = cv2.GaussianBlur(plateImg,(3,3),1.2)
        #plateImg[:,:] = int(plateImg[:,:] * 1.2)

        grayPlate = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)
        grayPlate = cv2.copyMakeBorder(grayPlate,2,2,2,2,cv2.BORDER_REPLICATE)
        plateImg = cv2.copyMakeBorder(plateImg,2,2,2,2,cv2.BORDER_REPLICATE)
        height, width = grayPlate.shape
        cv2.imshow('grayPlate',grayPlate)
        # Blur image
        # grayPlate = cv2.GaussianBlur(grayPlate, (5, 5), 1.0) # remove noise
        #_, threshImg = cv2.threshold(grayPlate, 80, 255, cv2.THRESH_BINARY_INV)
        threshImg = cv2.adaptiveThreshold(grayPlate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,7,2)
        cv2.imshow('Thresh',threshImg)

        # Opening for removing noise
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
        openImg = cv2.morphologyEx(threshImg,cv2.MORPH_OPEN,kernel)
        cv2.imshow('Opening (erosion followed by dilation)',openImg)

        # Erode plate image for removing noise
        '''kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
        erodeImg = cv2.erode(threshImg, kernel, iterations = 1)
        cv2.imshow('Erode', erodeImg)'''

        # Dilate plate image for OCR
        '''kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        dilateImg = cv2.dilate(openImg, kernel, iterations = 1)
        cv2.imshow('Dilate',dilateImg)'''
        # cv2.imshow('Dilate',dilateImg)
        # Threshold image
        #_, threshImg = cv2.threshold(dilateImg, 150, 255, cv2.THRESH_BINARY)
        #cv2.imshow('Thresh',threshImg)
        # cv2.imshow('Threshold',threshImg)
        #ret, threshImg = cv2.threshold(dilateImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Find contours
        
        contours, hierachy = cv2.findContours(np.copy(openImg), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # get corner points on boundary
        
        tmpImg = plateImg.copy()
        cv2.drawContours(tmpImg,contours,-1,(0,255,0),1)
        cv2.imshow('Char Contours',tmpImg)
        cv2.imwrite('con.jpg',tmpImg)
        # If contours exists
        print('Number of contours: ', len(contours))
        if len(contours) > 7:
            # Compute Area of each contour (Moment M[00])
            area_list = [cv2.contourArea(cnt) for cnt in contours]
            
            index = []
            x,y,w,h = (0,0,0,0)
            for (i,cnt) in enumerate(contours):
                # Get bouding rect of this contour
                x, y, w, h = cv2.boundingRect(cnt)
                # Verify sub image
                '''if i != 9:
                    continue'''
                # Crop region that have number in image
                #found_plate = np.copy(grayPlate[y:y+h, x:x+w])
                
                if y - 2 >= 0 and x - 2 >= 0 and y+h+2 < height and x+w+2 < width:
                    found_plate = np.copy(grayPlate[y - 2:y + h + 2, x - 2:x + w + 2])
                else:
                    found_plate = np.copy(grayPlate[y:y + h, x:x + w])
                
                # found_plate = cv2.copyMakeBorder(found_plate,2,2,2,2,cv2.BORDER_CONSTANT,value=[255,255,255])
                #print('Check size')
                '''tmpImg = plateImg.copy()
                cv2.drawContours(tmpImg,contours,i,(0,255,0),1)
                cv2.imshow('Char Contour' + str(i),tmpImg)'''
                if self.verify_size_character(found_plate):
                    #print(str(i))
                    '''tmpImg = plateImg.copy()
                    cv2.drawContours(tmpImg,contours,i,(0,255,0),1)
                    cv2.imshow('Char Contour' + str(i),tmpImg)'''
                    #cv2.imshow('Char' + str(i), found_plate)
                    index.append(x)
                    ret.append((found_plate,(x,y,w,h)))


            indices = np.argsort(index)
            ret = [x for _,x in sorted(zip(index,ret))]
            
            return ret
        
        return None