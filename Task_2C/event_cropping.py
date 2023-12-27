# Import packages
import cv2
import numpy as np

img = cv2.imread('sample.png')
cv2.imshow("original", img)

# Cropping an image
images = [img[120:170, 160:210],
img[340:390, 150:200],
img[340:390, 470:520],
img[470:520, 460:510],
img[600:650, 160:210]]
#
# for i in range(len(images)):
#     # cv2.imshow("original", image)
#     cv2.imwrite(f"{i}.png", images[i])
# #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
reversed_images = images[::-1]
print(reversed_images)