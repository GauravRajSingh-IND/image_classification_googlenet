import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# image path.
path = "panda.jpg"

# read image.
image = cv.imread(path, cv.IMREAD_COLOR)

# read model.
weight_file = "bvlc_googlenet.caffemodel"
proto_file = "bvlc_googlenet.prototxt"

# read all the classes from txt file and store in a variable..
with open('classification_classes_ILSVRC2012.txt', 'r') as f:
    classes = f.read().splitlines()


# Display the image.
plt.imshow(image[:,:,::-1])
plt.title("Image of a Panda")
plt.axis('off')
plt.show()