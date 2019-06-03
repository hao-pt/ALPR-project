import numpy as np
import cv2
from pytesseract import image_to_string
import matplotlib.pyplot as plt
import chars_separation

class CharactersRecognition:

    def __init__(self):
        self.init = 0


    def recognize_characters(self, listOfPlates, src_img):
        for (i, found_plate) in enumerate(listOfPlates):
            '''if i == 1:
                break'''
            
            #cv2.imshow('Found plate', found_plate)
            # Take advantage of white background, so we will check if the average color >= 100, it will a plate
            if np.mean(found_plate).astype(np.uint8) >= 100:
                # Display plate
                plt.figure()
                plt.imshow(cv2.cvtColor(found_plate, cv2.COLOR_BGR2RGB))
                plt.title("Correct plates")
                plt.axis("off")

                plateNumber = ""

                charsSeparator = chars_separation.CharactersSeparation()
                charImgs = charsSeparator.get_character_images(found_plate)
                if charImgs is None:
                    continue
                #print(len(charImgs))
                config = ("-l eng --oem 1 --psm 7")
                old_x,old_y,old_w,old_h = -1,-1,-1,-1 # for solving "O" or "o" or "0", which have 2 contours
                for (i,charInf) in enumerate(charImgs):
                    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    #closingImg = cv2.morphologyEx(charImg, cv2.MORPH_CLOSE, kernel)
                    x,y,w,h = charInf[1]
                    if x - old_x < 8:
                        continue
                    c_height, c_width = charInf[0].shape
                    #charImg = cv2.resize(charImg, (c_width*2,int(c_height*1.5)))
                    

                    charImg = cv2.copyMakeBorder(charInf[0],2,2,2,2,cv2.BORDER_REPLICATE)
                    charImg = cv2.adaptiveThreshold(charImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,5)
                    charImg = cv2.GaussianBlur(charImg,(3,3),1.5)
                    '''cv2.imshow('new size: ' + str(i),charImg)

                    threshPlate = cv2.adaptiveThreshold(charImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,5)
                    cv2.imshow('Thresh: '+ str(i), threshPlate)

                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                    openImg = cv2.morphologyEx(threshPlate,cv2.MORPH_OPEN,kernel)


                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    #eroseImg = cv2.erode(threshPlate, kernel, iterations = 1)
                    dilateImg = cv2.dilate(openImg, kernel, iterations = 1)'''
                    
                    old_x,old_y,old_w,old_h = x,y,w,h

                    #cv2.imshow('Thresh char' + str(i), threshPlate)
                    cv2.imshow('Dilate char' + str(i), charImg)
                    char = image_to_string(charImg, config = config)
                    if len(char) > 0:
                        char = char[0]
                    char = char.replace(" ", "")
                    print(char,' ', x)
                    plateNumber += char
                print(plateNumber)
                

                Height, Weight = src_img.shape[:2]
                # Specify font
                font = cv2.FONT_HERSHEY_SIMPLEX
                # draw text
                cv2.putText(src_img, plateNumber, (int(Weight/2) - 100, int(Height/2)), font, 2, (0,0,255), 2, cv2.LINE_AA)
                
                plt.figure()
                plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
                plt.title("Plate number recognition")
                plt.axis("off")

                #print(plate_text)
                return plateNumber

        print("None")
        return "None"



# ==================================NOT SEPARATING CHARACTER===========================================
# Convert to gray image
'''grayPlate = cv2.cvtColor(found_plate, cv2.COLOR_BGR2GRAY)
height,width = grayPlate.shape
# increasing size of the image
grayPlate = cv2.resize(grayPlate,(2*width,2*height))
cv2.imshow('Gray', grayPlate)
# Blur image
blurPlate = cv2.GaussianBlur(grayPlate, (5, 5), 1.2) # Remove noise
# Apply closing operation:  Dilation followed by Erosion to remove noise and hole inside object
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
closingImg = cv2.morphologyEx(blurPlate, cv2.MORPH_CLOSE, kernel)
cv2.imshow('morpho',closingImg)
# Apply threshold to get image with only b&w (binarization)
#_, threshPlate = cv2.threshold(closingImg, 120, 255, cv2.THRESH_BINARY) #+ cv2.THRESH_OTSU)
threshPlate = cv2.adaptiveThreshold(closingImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
#v2.imshow('Thresh',threshPlate)
# Store image in PIL image format
pilImg = Image.fromarray(threshPlate)
pilImg = np.array(pilImg)
cv2.imshow('pilImg',pilImg)
cv2.imwrite('test1.jpg',pilImg)
# Recognize text with Tesseract
plate_text = image_to_string(pilImg, lang = "eng")
plate_text = plate_text.replace(" ", "") # Remove space

if len(plate_text) == 0:
    plate_text = "None"
                    
# Print Recognize text on image
copy_img = np.copy(img)
# Specify font
font = cv2.FONT_HERSHEY_SIMPLEX
# draw text
cv2.putText(copy_img, plate_text, (x - w, y + 4*h), font, 2, (0,0,255), 2, cv2.LINE_AA)
# Draw contour
cv2.drawContours(copy_img, [cnt], 0, (0, 255, 0), 2)

plt.figure()
plt.imshow(cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB))
plt.title("Plate number recognition")
plt.axis("off")'''
