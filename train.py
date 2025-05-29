import os
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'mask_detector.keras')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create necessary directories
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_data_pipeline():
    """Create optimized data pipeline with class verification"""
    # Load datasets without prefetch first to get class names
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        label_mode='binary',
        image_size=(100, 100),
        batch_size=32,
        seed=42,
        color_mode='rgb'
    )

    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        label_mode='binary',
        image_size=(100, 100),
        batch_size=32,
        seed=42,
        color_mode='rgb'
    )

    # Get class names before prefetch
    class_names = raw_train_ds.class_names
    print("\nDataset class indices:")
    print(f"Training classes: {class_names}")
    print(f"Validation classes: {raw_val_ds.class_names}")

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model():
    """Create CNN model with proper initialization"""
    return Sequential([
        Input(shape=(100, 100, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])


def save_training_plots(history):
    """Save accuracy/loss visualization"""
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training metrics to {plot_path}")


def visualize_predictions(model, dataset, class_names):
    """Verify predictions with sample images"""
    plt.figure(figsize=(15, 10))
    images, labels = next(iter(dataset.take(1)))

    predictions = model.predict(images)
    batch_size = images.shape[0]

    for i in range(min(9, batch_size)):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].numpy().astype("uint8")
        true_idx = int(labels[i] > 0.5)
        pred_idx = int(predictions[i] > 0.5)

        plt.imshow(img)
        plt.title(
            f"True: {class_names[true_idx]}\n"
            f"Pred: {class_names[pred_idx]}\n"
            f"({predictions[i][0]:.2f})",
            color='green' if true_idx == pred_idx else 'red'
        )
        plt.axis("off")

    plot_path = os.path.join(RESULTS_DIR, 'sample_predictions.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved sample predictions to {plot_path}")


def main():
    print(f"\nMask Detection Training")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Dataset path: {DATASET_PATH}")

    # Data pipeline with class names
    train_ds, val_ds, class_names = build_data_pipeline()

    # Model configuration
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Training
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=1
    )

    # Save and visualize
    model.save(MODEL_PATH)
    save_training_plots(history)
    visualize_predictions(model, val_ds, class_names)

    print(f"\nTraining complete!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class order: {class_names}")


if __name__ == "__main__":
    main()