import cv2
import os
import numpy as np
def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset")

    model = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    print("Total faces collected:", len(faces))

    if len(faces) == 0:
        print("ERROR: No training images found!")
        exit()

    faces = np.array(faces)
    labels = np.array(labels)

    model.train(faces, labels)

    model.save(os.path.join(BASE_DIR, "face_model.yml"))
    np.save(os.path.join(BASE_DIR, "labels.npy"), label_map)

    print("Training successful!")
    print("Files created:")
    print("✔ face_model.yml")
    print("✔ labels.npy")
