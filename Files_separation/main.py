import os
import argparse
import numpy as np
import cv2
import plate_detection
import chars_separation
import chars_recognition


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to input image")
ap.add_argument("-o", "--option", required= False, help = "Show step by step", default = "0") 
args = vars(ap.parse_args())

def main(args):
    # Measure time
    e1 = cv2.getTickCount()
    
    # bool to show step by step
    isShow = True if args["option"] == "1" else False

    img = cv2.imread(args["input"])  # ".\\test_images\\FA600CH.png"

    # Plate Detection
    plateDetector = plate_detection.PlateDetection()
    listOfPlates = plateDetector.detect_plates(img, step_by_step = isShow)
    # Characters Recognition
    charactersRecognitor = chars_recognition.CharactersRecognition()
    plateNumbers = charactersRecognitor.recognize_characters(listOfPlates)

    # End time
    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print('Time: %.2f(s)' %(time))

    #plt.show()
    cv2.waitKey()

if __name__ == "__main__":
    main(args)