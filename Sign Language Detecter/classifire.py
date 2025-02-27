import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('data.pkl', 'rb'))

required_features = 84
uniform_data = np.array([np.pad(x, (0, required_features - len(x)), 'constant') if len(x) < required_features else x[:required_features] for x in data_dict['data']])

data = np.asarray(uniform_data)
labels = np.asarray(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred)*100)

f = open('model.pkl', 'wb')
pickle.dump({'model': clf}, f)
f.close()