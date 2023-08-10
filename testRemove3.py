import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
from  PIL  import Image





img = cv.imread('test_data3\inputs\magaye.jpg', cv.IMREAD_UNCHANGED)

mask = cv.imread("test_data3/masks/magaye.png")

# mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

output = np.zeros((mask.shape[0], mask.shape[1], 4))

alpha= np.zeros((mask.shape[0], mask.shape[1], 1))
imgRgba = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
# imgRgba[:,:,3] =alpha[:,:,0]
mask = cv.cvtColor(mask, cv.COLOR_BGR2BGRA)

# mask = cv.bitwise_( mask)

cnd = mask[:, :, 1] > 30

print(np.amax(mask))

# cnd = [cnd0 and cnd1 and cnd2]
output[cnd]= imgRgba[cnd]



# imgCropped = imgRgba[46:119,352:495]

# cv.imwrite('mask.png', mask)

# result = cv.bitwise_and(imgRgba, mask)

# result[mask ==  0] =  255

cv.imshow('img', img)
# cv.imshow('imgRgba', imgRgba)
cv.imshow('mask', mask)
# cv.imshow('output', output[:,:,3])
# cv.imshow('imgCropped', imgCropped)
cv.imwrite('bg2.png', output)



cv.waitKey(0)
cv.destroyAllWindows()