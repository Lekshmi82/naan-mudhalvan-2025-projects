import cv2
import numpy as np
import joblib
import sys

def extract_orb_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is not None:
        return descriptors.flatten()[:512]
    else:
        return np.zeros(512)

def predict_note(image_path, model_path="note_auth_model.pkl"):
    model = joblib.load(model_path)
    features = extract_orb_features(image_path)
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction]
    label = "Genuine" if prediction == 1 else "Counterfeit"

    image = cv2.imread(image_path)

    text = f"{label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (30, 50), font, 1, (0, 255, 0) if label == "Genuine" else (0, 0, 255), 2)

    cv2.imshow("Currency Note Authentication", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"[RESULT] The currency note is: {label} ")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test/3.jpg" #put test image
    predict_note(image_path)