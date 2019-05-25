import cv2
import os
import matplotlib.pyplot as plt
from google.cloud import vision_v1p3beta1 as vision

# Set up google authetic key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud-vision.json'

def recognize_license_plate(img_path):
    # Measure time
    e1 = cv2.getTickCount()

    # Read image
    img = cv2.imread(r".\Bike_back\1.jpg")
    
    # Get size of image
    height, width = img.shape[:2]

    # Resize image: width = 800, height = (original_height * 800 / orignal_width
    img = cv2.resize(img, (800, round(height*800/width)))

    # Show original image
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Because matplotlib use RGB mode
    plt.title("Original image"); plt.axis("off")

    # Get path
    parts = img_path.split("\\")
    path = "\\".join(parts[:-1])
    
    # Create path to resized image
    resized_img_path = path + "\\" + "resized_img.jpg"
    print(resized_img_path)
    # Save image as temporary file to process
    cv2.imwrite(resized_img_path, img)

    # Create google vision client object
    client = vision.ImageAnnotatorClient()

    # Read image as binary file
    with open(resized_img_path, "rb") as f:
        img_content = f.read()

    image = vision.types.Image(content = img_content)

    # Recognize text
    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        if len(text.description) == 10:
            license_plate = text.description
            print(license_plate)
            vertices = [(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices]

            # Put text license plate number to image
            cv2.putText(img, license_plate, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            print(vertices)
            # Draw rectangle around license plate
            cv2.rectangle(img, (vertices[0][0]-10, vertices[0][1]-10), (vertices[2][0]+10, vertices[2][1]+10), (0, 255, 0), 3)
                        
            # Display
            plt.figure()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Because matplotlib use RGB mode
            plt.title("Recognize plate number"); plt.axis("off")

    # End time
    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print('Time: %.2f(s)' %(time))

    plt.show()

recognize_license_plate(".\\Bike_back\\1.jpg")
