import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv

from utils import *
from segment import *

image_path = './images/land.jpg'
image_size = (256, 256)

# Load image
image = cv.imread(image_path)
# Convert to rgb 
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# Resize image to image_size
image = cv.resize(image, image_size)
# Show image
# plt.imshow(image)
# plt.show()

# Do a segmantation for Gray-Scale Image
gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
G_gray = segmentation(gray_image, sigma=4., neigh=1, K=image.shape[0])
segmented_image = viz_segmentation(gray_image, G_gray, gray=True, display=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].imshow(image)
axes[0].set_title('Original image')
axes[1].imshow(segmented_image)
axes[0].set_title('Segmented image')

# save segmented image
plt.savefig('./results/land_segmented.png')
plt.show()