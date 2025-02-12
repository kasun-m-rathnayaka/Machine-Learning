import cv2, os

img = cv2.imread(os.path.join('.', 'data', 'img.jpg'))

# line
cv2.line(img, (0, 0), (1000, 1000), (100, 255, 0), 3)

# rectangle
cv2.rectangle(img, (100, 120), (400, 600), (255, 0, 0), -1)

# circle
cv2.circle(img, (300, 300), 100, (0, 0, 255), 3)

# text
cv2.putText(img, "This is sample text", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

# show images
cv2.imshow('img', img)
cv2.waitKey(0)
