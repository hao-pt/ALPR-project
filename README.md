# ALPR-project

## Introduction:
University of Sciences

Subject: Introduction to Computer Vision

Final project: Automatic License Plate Recognition (ALPR)

We develop 2 versions:
- Version 1: Process for car plate in Europe. Almost based on handcraft algorithm to segment region in image and then extract license plate region. Also we use SVM classifer to class image as Plate or Not. Finally, we use Tesseract library to recognize plate numbers. But I find Tesseract didn't work well with recognizing each single plate character/number, Tesseract just works well with recognizing semantic sentences. Because it use Long Short-Term Memory (LSTM).
- Version 2: Process for motor plate in Viet Nam. I use transfer learning with pretrain model from https://thigiacmaytinh.com/modelcascade-da-huan-luyen/. We use Haar-Cascade to extract Haar feature and then use it for detecting license plate region.

More detail, please check it at our proposal and report document.
Those 2 doesn't give the best perfomance to compare with state-of-the-art project in related field. They are just acceptable results. 

## Team Fusion:
1. 1612174 Phùng Tiên Hao tienhaophung@gmail.com
2. 612269 Võ Quôc Huy voquochuy304@gmail.com
3. 1612272 Trân Nhât Huy nhathuy13598@gmail.com

## Documents:
There are proposal and report file to have you can refer them and get more intuitive. These document was written in Vietnamese.



