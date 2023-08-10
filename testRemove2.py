import cv2
import numpy as np

file_name = "test_data3/inputs/magaye.jpg"
mask_file = "test_data3/masks/magaye.png"

        

def transBg(img):   
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

#   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
#   morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
from  PIL  import Image

img = cv.imread('images/test.jpg', cv.IMREAD_UNCHANGED)
original = img.copy()

l = int(max(5, 6))
u = int(min(6, 6))

ed = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.GaussianBlur(img, (21, 51), 3)
edges = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
edges = cv.Canny(edges, l, u)

_, thresh = cv.threshold(edges, 0, 255, cv.THRESH_BINARY  + cv.THRESH_OTSU)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)

data = mask.tolist()
sys.setrecursionlimit(10**8)
for i in  range(len(data)):
    for j in  range(len(data[i])):
        if data[i][j] !=  255:
            data[i][j] =  -1
        else:
            break
    for j in  range(len(data[i])-1, -1, -1):
        if data[i][j] !=  255:
            data[i][j] =  -1
        else:
            break
image = np.array(data)
image[image !=  -1] =  255
image[image ==  -1] =  0

mask = np.array(image, np.uint8)

result = cv.bitwise_and(original, original, mask=mask)
result[mask ==  0] =  255
cv.imwrite('bg.png', result)

img = Image.open('bg.png')
img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] ==  255  and item[1] ==  255  and item[2] ==  255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save("img.png", "PNG")
#   roi, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#   mask = np.zeros(img.shape, img.dtype)
    mask = cv2.imread(f'test_data3/masks/{image}.png' )

  cv2.fillPoly(mask, roi, (255,)*img.shape[2], )

  masked_image = cv2.bitwise_and(img, mask)

  return masked_image

def fourChannels(img):
  height, width, channels = img.shape
  if channels < 4:
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return new_img

  return img

s_img = cv2.imread(file_name, -1)

# set to 4 channels
s_img = fourChannels(s_img)

# remove white background
s_img = cut(s_img)

# set background transparent
s_img = transBg(s_img)

cv2.imwrite("test_data3/results/magaye2.png", s_img)