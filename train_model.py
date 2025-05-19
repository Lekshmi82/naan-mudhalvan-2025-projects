import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def extract_orb_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is not None:
        return descriptors.flatten()[:512]  # Limit to fixed length
    else:
        return np.zeros(512)

def load_data(dataset_path):
    X, y = [], []
    for label in ['genuine', 'counterfeit']:
        folder_path = os.path.join(dataset_path, label)
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            features = extract_orb_features(path)
            X.append(features)
            y.append(1 if label == 'genuine' else 0)
    return np.array(X), np.array(y)

X, y = load_data("dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

print("[INFO] Training complete. Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "note_auth_model.pkl")
print("[INFO] Model saved as note_auth_model.pkl")