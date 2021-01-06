import cv2 as cv
from matplotlib import pyplot as plot

img = cv.imread("plant@2x.jpg", 0)

# extreme examples: high threshold [900, 1000]; low threshold [1, 10]
edges = cv.Canny(img, 50, 100)

# sets the plot's title
plot.title("Canny Edges")

# removes the numbers from the x and y axis
plot.xticks([]), plot.yticks([])

# shows the final plot
plot.imshow(edges, cmap="gray")
plot.show()