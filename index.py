import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv
import argparse

from utils import *
from segment import *

parser = argparse.ArgumentParser(description='Image Segmentation')
parser.add_argument('--image_path', type=str, default='./images/lena.png', help='path to image')
parser.add_argument('--image_size', type=int, default=128, help='image size')
parser.add_argument('--sigma', type=float, default=5., help='sigma')
parser.add_argument('--neigh', type=int, default=2, help='neigh')
parser.add_argument('--K', type=int, default=128, help='K')
args = parser.parse_args()

image_name = args.image_path.split('/')[-1].split('.')[0]
image_size = (args.image_size, args.image_size)

# Load image
image = cv.imread(args.image_path)
# Convert to rgb 
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# Resize image
image = cv.resize(image, image_size)

G = segmentation(image, sigma=args.sigma, neigh=args.neigh, K=args.K)
segmented_image = viz_segmentation(image, G, display=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].imshow(image)
axes[0].set_title('Original image')
axes[1].imshow(segmented_image)
axes[0].set_title('Segmented image')

# save segmented image
plt.savefig(f'./results/{image_name}_segmented_{args.image_size}_n{args.neigh}_sigma{args.sigma}.png')
plt.show()