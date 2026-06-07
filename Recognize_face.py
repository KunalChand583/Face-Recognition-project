import cv2
import numpy as np
import os


def recognize():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
    LABELS_PATH = os.path.join(BASE_DIR, "labels.npy")

   
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("❌ Model not found. Please train first.")
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)

    label_map = np.load(LABELS_PATH, allow_pickle=True).item()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            label, confidence = model.predict(face)

            # ✅ CONFIDENCE THRESHOLD (CRITICAL)
            if confidence > 70:   # adjust 60–90 if needed
                name = "Unknown"
            else:
                name = label_map.get(label, "Unknown")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({round(confidence, 2)})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
