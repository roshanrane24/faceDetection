import cv2
import numpy as np
import os

def detectFaces(image, threshold=0.5, video=False):
    '''Detect faces from input images & return image with bounding boxes & confidence'''

    # Check if model files are available & if not available ask for file path
    if os.path.exists('deploy.prototxt.txt'):
        prototxt = 'deploy.prototxt.txt'
    else:
        prototxt = str(input("Path to prototxt file:\n"))


    if os.path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
        model = 'res10_300x300_ssd_iter_140000.caffemodel'
    else:
        model = str(input("Path to caffemodel file:\n"))

    # Create caffe model from input files
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Read input image
    if not video:
        img = cv2.imread(image)
    else:
        img = image
    # Extract height & width of image
    (height, width) = img.shape[:2]
    # Create a image blob
    blob_shape = (300, 300)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, blob_shape), 1.0, blob_shape, (104.0, 177.0, 123.0))
    count = 0
    # Compute object detection
    net.setInput(blob)
    predictions = net.forward()

    # looping over detections
    for region in range(predictions.shape[2]):
        # extract confidence for current detect region
        confidence = predictions[0, 0, region, 2]

        # select region with prediction higher than threshold
        if confidence > threshold:
            # compute coordinates of bounding box
            bbox = predictions[0, 0, region, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = bbox.astype("int")
            # probability text
            text = f"{round(confidence * 100, 2)} %"
            text_y = y1 + 10 if (y1 + 10 < 10) else y1 - 10

            # draw bounding box on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 164, 0), 1)
            count += 1
    return img, count
