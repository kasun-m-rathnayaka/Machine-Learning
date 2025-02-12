import cv2, os

img = cv2.imread(os.path.join('.','data','img.png'))

# blur
k_size = 7
img_blur = cv2.blur(img,(k_size,k_size))
img_gaussianblur = cv2.GaussianBlur(img,(k_size,k_size),3)
img_mediamblur = cv2.medianBlur(img,k_size)

# show images
cv2.imshow('img',img)
cv2.imshow('img_blur',img_blur)
cv2.imshow('img_mediamblur',img_mediamblur)
cv2.imshow('img_gaussianblur',img_gaussianblur)
cv2.waitKey(0)
