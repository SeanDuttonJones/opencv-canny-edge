import cv2 as cv
from matplotlib import pyplot as plot

# helper function to easily display our images
def img_show(title, image):
    plot.title(title)
    plot.xticks([])
    plot.yticks([])
    plot.imshow(image, cmap="gray")
    plot.show()

# read in our original image as grayscale
img = cv.imread("original@2x.jpg", cv.IMREAD_GRAYSCALE)

# show grayscale image using our helper function
img_show("Grayscale Image", img)
cv.imwrite("grayscale.jpg", img)

# blurring the image with a 5x5, sigma = 1 Guassian kernel
img_blur = cv.GaussianBlur(img, (5, 5), 1)

img_show("Gaussian Blur", img_blur)
cv.imwrite("gaussian_blur.jpg", img_blur)

# applying both horizontal and vertical 3x3 Sobel kernels
img_sobelx = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=3)
img_sobely = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=3)

img_show("Horizontal(x) Sobel", cv.convertScaleAbs(img_sobelx))
img_show("Vertical(y) Sobel", cv.convertScaleAbs(img_sobely))
cv.imwrite("sobel_x.jpg", img_sobelx)
cv.imwrite("sobel_y.jpg", img_sobely)

# finally, generate canny edges
# extreme examples: high threshold [900, 1000]; low threshold [1, 10]
img_edges = cv.Canny(img, 50, 100)

img_show("Canny Edges", img_edges)
cv.imwrite("canny_edges.jpg", img_edges)