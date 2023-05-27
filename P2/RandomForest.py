from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image

dataset_path = 'dataset_256X256'

train_data = []
train_labels = []

for label in os.listdir(os.path.join(dataset_path, 'train')):
    label_folder = os.path.join(dataset_path, 'train', label)
    for image_file in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_file)
        image = Image.open(image_path)
        train_data.append(np.array(image))
        train_labels.append(label)


test_data = []
test_labels = []

for label in os.listdir(os.path.join(dataset_path, 'test')):
    label_folder = os.path.join(dataset_path, 'test', label)
    for image_file in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_file)
        image = Image.open(image_path)
        test_data.append(np.array(image))
        test_labels.append(label)

# Convert lists to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

X_train_flat = train_data.reshape(train_data.shape[0], -1)
X_val_flat = test_data.reshape(test_data.shape[0], -1)

clf = RandomForestClassifier()
clf.fit(X_train_flat, train_labels)

y_pred = clf.predict(X_val_flat)

accuracy = accuracy_score(test_labels, y_pred)
print("Validation Accuracy:", accuracy)
