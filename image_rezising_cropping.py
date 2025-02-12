import cv2, os

img = cv2.imread(os.path.join('.','data','img.jpg'))

# resize
resized_img = cv2.resize(img,(640, 480))
print(img.shape)
print(resized_img)

# crop image
cropped_image = img[320:640, 420:840]

# show images
cv2.imshow('img',img)
cv2.imshow('cropped_image',cropped_image)
cv2.waitKey(0)
