import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def test_single_image(image_path, model_path):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (100, 100)) / 255.0
    pred = model.predict(np.expand_dims(resized, axis=0))[0][0]

    plt.imshow(img)
    plt.title(f"Mask Detected: {'YES' if pred > 0.5 else 'NO'} ({pred:.2f})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_single_image(
        image_path='D:/hermon/damdam/maskdetection/dataset/with_mask/with_mask_100.jpg',
        model_path='D:/hermon/damdam/maskdetection/model/mask_detector.keras'
    )