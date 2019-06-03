import os
import argparse
import numpy as np
import cv2
import plate_detection
import chars_separation
import chars_recognition
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to input image")
ap.add_argument("-o", "--option", required= False, help = "Show step by step", default = "0") 
args = vars(ap.parse_args())

def classify_plate(plate_img):
    tree = ET.parse(r'E:\K16\Junior\TGMT\ALPR-project\Files_separation\SVM.xml')  
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


def main(args):
    # Measure time
    e1 = cv2.getTickCount()
    
    # bool to show step by step
    isShow = True if args["option"] == "1" else False

    img = cv2.imread(args["input"])  # ".\\test_images\\FA600CH.png"

    # Plate Detection
    plateDetector = plate_detection.PlateDetection()
    eqhist_plates, plate_images = plateDetector.detect_plates(img, step_by_step = isShow)

    # List of correct plate
    listCorrectPlate = []

    # Classify
    for i, plate_image in enumerate(eqhist_plates):
        if classify_plate(plate_image):
            listCorrectPlate.append(plate_images[i])

    # Characters Recognition
    charactersRecognitor = chars_recognition.CharactersRecognition()
    plateNumbers = charactersRecognitor.recognize_characters(listCorrectPlate, img.copy())

    # End time
    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print('Time: %.2f(s)' %(time))

    plt.show()
    cv2.waitKey()

if __name__ == "__main__":
    main(args)