import cv2, os

img = cv2.imread(os.path.join('.', 'data', 'img.jpg'))

# global threshold
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray,80,255, cv2.THRESH_BINARY)
img_blur =cv2.blur(thresh,(10,10))
ret, thresh = cv2.threshold(thresh,80,255, cv2.THRESH_BINARY)

# adaptive threshold
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)

# show images
cv2.imshow('img',img)
cv2.imshow('abp_thresh',adaptive_thresh)
cv2.waitKey(0)
