import numpy as np
import cv2

# Use to find bounding box of textbox and corresponding confidences 
def decode(scores, geometry, conf_thresh = 0.5):
    rows, cols = scores.shape[2:4]
    # boudingBox of textbox
    boundingBox = []
    confidences = []

    # Loop each row
    for r in range(rows):
        # Get coordinate x of 4 corner of boudary
        x1 = geometry[0, 0, r]
        x2 = geometry[0, 1, r]
        x3 = geometry[0, 2, r]
        x4 = geometry[0, 3, r]
        # Get angle of rotated rect
        angles = geometry[0, 4, r]
        # Get confidence score of this textbox
        confs = scores[0, 0, r]

        # Loop over column to make bouding box
        for c in range(cols):
            # Check if confidences < conf_thresh
            if confs[c] < conf_thresh:
                continue
            
            # Compute width and height of textbox
            boxH = x1[c] + x3[c]
            boxW = x2[c] + x4[c]

            # Compute offset factor because feature maps will be 4x smaller than input image
            (offsetX, offsetY) = (r*4.0, c*4.0)

            # Angle and compute cos, sine for affine rotation
            angle = angles[c]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Compute Top-left corner and bottom-left corner
            endX = int(offsetX + (cos * x2[c]) + (sin * x3[c]))
            endY = int(offsetY - (sin * x2[c]) + (cos * x3[c]))
            startX = int(endX - boxW)
            startY = int(endY - boxH)

            # store bounding box
            boundingBox.append((startX, startY, endX, endY))
            confidences.append(confs[c])

    return boundingBox, confidences

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

# Read image
img = cv2.imread(".\\test_images\\2715DTZ.jpg")

# Get size of image
height, width = img.shape[:2]
# Resize image: To use EAST text require dimension of image are multiple of 32
img = cv2.resize(img, (320, 320))
# Get size of image again
height, width = img.shape[:2]

# Now we define 2 output layer name of EAST detector model
# Include:
#   Confidence of detected text in particular region
#   Coordinate of bouding box of text region, called "geometry"

layerName = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# Load pre-trained EAST model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Prepare imput image as 4-D blob to feed to network
blob = cv2.dnn.blobFromImage(img, \
                1.0, # scale factor
                (width, height),        # Size default: 320x320
                (123.68, 116.78, 103.94),   # Specify mean that will be subtracted from each image channel
                swapRB = True, crop = False) # Swap BGR to RGB and not crop image

# Forward pass through the network
# To get 2 output layer of network: geometry of textbox and confidence score of the detected box
net.setInput(blob)
(scores, geometry) = net.forward(layerName)

# Process the output: geometry and confidence scores
# Decode the positions of the text boxes
# geomtry include: Top-left and bottom-right coordinates and angle
[boxes, confidences] = decode(scores, geometry, 0.5)

# Remove overplapping box with non-maximum surpression
# Apply NMS with conf_thresh = 0.5, nms_thresh = 0.4
boxes = non_max_suppression_fast(np.array(boxes), confidences)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
 
	# draw the bounding box on the image
	cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
 
# show the output image
cv2.imshow("Text Detection", img)
cv2.waitKey(0)

print(indices)