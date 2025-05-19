import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model("model/food_model_three.h5")

# Convert to TFLite model with dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables quantization

# Convert the model
tflite_model = converter.convert()

# Save the .tflite model
with open("model/food_model_compressed.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion complete. File saved as food_model_compressed.tflite")
