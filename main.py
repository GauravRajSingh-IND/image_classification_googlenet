import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# image path.
path = "man.jpg"

# read image.
image = cv.imread(path, cv.IMREAD_COLOR)

# read model.
weight_file = "bvlc_googlenet.caffemodel"
proto_file = "bvlc_googlenet.prototxt"

# read all the classes from txt file and store in a variable..
with open('classification_classes_ILSVRC2012.txt', 'r') as f:
    classes = f.read().splitlines()

inHeight = 224
inWidth = 224
swap_rgb = False
mean = [104, 117, 123]
scale = 1.0

# load model.
net = cv.dnn.readNet(model=weight_file, config=proto_file, framework= "caffe")

# create a blob object
blob = cv.dnn.blobFromImage(image, scale, (inWidth, inHeight), mean, swap_rgb, crop=False)

# Run a model
net.setInput(blob)
out = net.forward()
out = out.flatten()

classID = np.argmax(out)
class_name = classes[classID]
confidence = f"{round(float(out[classID]) * 100, 2)}%"

print(class_name, confidence)

# Display the image.
plt.imshow(image[:,:,::-1])
plt.title(f"Image of a {class_name}, {confidence}")
plt.axis('off')
plt.show()