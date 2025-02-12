import cv2, os

img = cv2.imread(os.path.join('.','data','img.jpg'))

# convert color
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# show images
cv2.imshow('img',img)
cv2.imshow('img_rgb',img_rgb)
cv2.imshow('img_gray',img_gray)
cv2.imshow('img_hsv',img_hsv)
cv2.waitKey(0)
