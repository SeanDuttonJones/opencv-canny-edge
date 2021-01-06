import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("plant@2x.jpg", 0)

# extreme examples: high threshold [900, 1000]; low threshold [1, 10]
edges = cv.Canny(img, 50, 100)

# sets the plot's title
plt.title("Canny Edges")

# removes the numbers from the x and y axis
plt.xticks([]), plt.yticks([])

# shows the final plot
plt.imshow(edges, cmap="gray")
plt.show()