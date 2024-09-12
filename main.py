import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# image path.
path = "panda.jpg"

# read image.
image = cv.imread(path, cv.IMREAD_COLOR)

# Display the image.
plt.imshow(image[:,:,::-1])
plt.title("Image of a Panda")
plt.axis('off')
plt.show()