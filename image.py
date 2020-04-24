from detectFaces import detectFaces
import argparse
import cv2
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to image.")
args = vars(parser.parse_args())

# Detect faces from image
image, count  = detectFaces(args["image"])
print(f'detected {count} faces')
# output image
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)