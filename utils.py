import os
import cv2
import numpy as np


def load_test_images(dataset_path):
    categories = ['with_mask', 'without_mask']
    data = []
    labels = []

    for label, category in enumerate(categories):
        path = os.path.join(dataset_path, category)
        for img_name in os.listdir(path)[:50]:  # Sample 50 per class for testing
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (100, 100))
                data.append(img)
                labels.append(label)
            except:
                continue

    X = np.array(data) / 255.0
    y = np.array(labels)
    return X, y