import os.path

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump

# prepare data
img2vec = Img2Vec()
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img in os.listdir(os.path.join(dir_, category)):
            img_path = os.path.join(dir_, category, img)
            # img_vec = img2vec.get_vec(img_path)
            img = Image.open(img_path).convert('RGB')

            img_features = img2vec.get_vec(img)
            features.append(img_features)
            labels.append(category)

        data[['training_data', 'validation_data'][j]] = features
        data[['training_labels', 'validation_labels'][j]] = labels

# Train model
model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])

# test performance
y_pred = model.predict(data['validation_data'])
score = accuracy_score(data['validation_labels'], y_pred)
print(score)

# save model
with open('model.pkl', 'wb') as f:
    dump(model, f)
    f.close()
