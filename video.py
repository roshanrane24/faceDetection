from detectFaces import detectFaces
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", default=0, help="Source of video")
args = vars(parser.parse_args())
capture = cv2.VideoCapture(int(args["source"]))

while True:
    state, frame = capture.read()
    if state == False:
        print("Check Source.")
        break
    image, count = detectFaces(frame, video=True)
    print(f"\r{count} faces detected", end='')
    cv2.imshow("Face detector", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()

