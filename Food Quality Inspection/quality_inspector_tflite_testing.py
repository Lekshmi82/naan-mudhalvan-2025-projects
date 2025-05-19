import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model/food_model_compressed.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
classes = ['A - Premium', 'B - Acceptable', 'C - Reject']

def preprocess(img):
    img = cv2.resize(img, (128, 128))  # Ensure it matches your training size
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------- TEST WITH STATIC IMAGE ----------
img_path = "testing/tomato3.jpg"
image = cv2.imread(img_path)
input_data = preprocess(image)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Show result
predicted_class = classes[np.argmax(output_data)]
print(f"Predicted Quality: {predicted_class}")

# Optional: display image with label
cv2.putText(image, f"Quality: {predicted_class}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()