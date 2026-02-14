import os
import cv2
import numpy as np

def load_dataset(dir_name, h=300, w=300):
    X = []
    y = []
    class_names = []
    person_id = 0
    n_samples = 0

    for person_name in os.listdir(dir_name):
        dir_path = os.path.join(dir_name, person_name)
        class_names.append(person_name)

        for image_name in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_name)

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray, (h, w))

            X.append(resized_image.flatten())
            y.append(person_id)
            n_samples += 1

        person_id += 1

    X = np.array(X)
    y = np.array(y)

    return X, y, class_names, n_samples
