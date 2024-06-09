import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_images_from_folder(folder):
    categories = ['rain', 'cloudy', 'shine', 'sunrise']
    images = []
    labels = []
    
    for category in categories:
        for filename in os.listdir(folder):
            if filename.startswith(category):
                img_path = os.path.join(folder, filename)
                img = load_img(img_path, target_size=(128, 128))  # Resize to desired size
                img = img_to_array(img)
                img = img / 255.0  # Normalize pixel values to [0, 1]
                
                images.append(img)
                labels.append(category)
    
    return np.array(images), np.array(labels)

folder = 'D:\\Feature Extraction\\weatherData'
images, labels = load_images_from_folder(folder)

#Converting categorical labels into numerical values 
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def extract_features(images, model):
    features = model.predict(preprocess_input(images))
    features = features.reshape(features.shape[0], -1)  # Flatten features
    return features

features = extract_features(images, model)

# Save features and labels
np.save('features.npy', features)
np.save('labels.npy', labels)

# Load for later use
# features = np.load('features.npy')
# labels = np.load('labels.npy')

# Modeling
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
