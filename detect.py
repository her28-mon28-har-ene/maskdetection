import cv2
import numpy as np
from tensorflow.keras.models import load_model

def enhance_face(image):
    """Preprocess face image identically to training"""
    # Convert to RGB and normalize
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (100, 100))
    normalized = resized.astype("float32") / 255.0
    return normalized

def detect_mask():
    model = load_model("../model/mask_detector.keras")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            try:
                # Enhanced preprocessing
                processed_face = enhance_face(face_roi)
                prediction = model.predict(np.expand_dims(processed_face, axis=0))[0][0]

                # Adaptive thresholding
                threshold = 0.65 if prediction > 0.5 else 0.35
                label = "With Mask" if prediction > threshold else "Without Mask"
                confidence = prediction if label == "With Mask" else 1 - prediction
                color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

                # Display confidence
                text = f"{label} ({confidence:.2%})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            except Exception as e:
                print(f"Processing error: {str(e)}")

        cv2.imshow("Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_mask()