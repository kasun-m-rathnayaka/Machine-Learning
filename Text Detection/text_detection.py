import pytesseract
from PIL import Image
import cv2
from easyocr import Reader

imge_path = "./data/study.jpg"
# text = pytesseract.image_to_string(Image.open(imge_path), lang='eng')
# print(text)

# cv2.imshow("Image", cv2.imread(imge_path))
# cv2.waitKey(0)

text = ' '
reader = Reader(['en'])
result = reader.readtext(imge_path)
for result in result:
    text = text + result[1] + " "
text = text[:-1]
print(text)