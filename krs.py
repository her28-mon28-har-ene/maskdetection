import tensorflow as tf
from tensorflow import keras

# Path to your .keras model file
model_path = 'D:/hermon/damdam/maskdetection/model/mask_detector.keras'

# Load the model
model = keras.models.load_model(model_path)

# Print basic model info
print("\n✅ Model Summary:")
model.summary()

# Print model input and output shapes
print("\n📥 Input shape:", model.input_shape)
print("📤 Output shape:", model.output_shape)

# Print each layer's name and output shape safely
print("\n🔍 Layers and Output Shapes:")
for layer in model.layers:
    try:
        print(f"{layer.name} --> {layer.output_shape}")
    except AttributeError:
        print(f"{layer.name} --> Output shape not available")

# Print weights count
weights = model.get_weights()
print("\n🧠 Number of weight arrays:", len(weights))

# Optionally print shapes of weights
print("\n📦 Shapes of weight arrays:")
for i, w in enumerate(weights):
    print(f"Weight {i}: {w.shape}")