from img2vec_pytorch import Img2Vec
from PIL import Image
from pickle import load
import os.path

with open('model.pkl', 'rb') as f:
    model = load(f)
    f.close()

img2vec = Img2Vec()
img_path = 'data/train/shine/shine16.jpg'
img = Image.open(img_path).convert('RGB')

features = img2vec.get_vec(img)

pred = model.predict([features])
print(pred)