from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())
model = load_model(args["model"])
labels = ["Bike", "Car", "Bus"]

imagePaths = sorted(list(paths.list_images(args["image"])))
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    # image = cv2.imread(args["image"])
    orig = image.copy()

    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    print("[INFO] loading network...")

    (Bike, Car, Bus) = model.predict(image)[0]


    # build the label
    # if Car > Bike:
    #     if Car > Bus:
    #         label = "Car"
    #     else:
    #         label = "Bus"
    # else:
    #     if Bike > Bus:
    #         label = "Bike"
    #     else:
    #         label = "Bus"
    label = np.argmax([Bike, Car, Bus])
    label = labels[label]
    # label = "Car" if Car > Bike else "Bike"
    proba = max([Bike, Car, Bus])
    label = "{}: {:.2f}%".format(label, proba * 100)
    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.imshow("Output", output)
    cv2.waitKey(0)


