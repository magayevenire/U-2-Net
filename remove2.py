import cv2
import numpy as np

# Read image
img = cv2.imread('test_data/test_images/horse.jpg')
mask = cv2.imread('test_data/u2net_results/horse.png' )
mask2 = cv2.imread('test_data/u2netp_results/horse.png' )


final = cv2.bitwise_and(img, mask)
final2 = cv2.bitwise_and(img, mask2)

print(mask.shape)
cv2.imshow('img', img)
cv2.imshow('mask', mask)
cv2.imshow('mask2', mask2)

cv2.imshow('final', final)
cv2.imshow('final2', final2)
# cv2.imwrite('final.png', final)

cv2.waitKey(0)
cv2.destroyAllWindows()