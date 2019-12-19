import numpy as np
import cv2
import os, sys
from tqdm import tqdm


if __name__ == '__main__':
    clf_cad = cv2.CascadeClassifier()
    face_cascade_name = 'haarcascade_frontalface_alt.xml'
    clf_cad.load(face_cascade_name)

    data_dir = './output'
    faces_path = os.listdir(data_dir)
    faces_path = [os.path.join(data_dir, x) for x in faces_path]

    total_seen = 0
    correct = 0
    for path in tqdm(faces_path):
        face = cv2.imread(path)
        if face is None:
            continue
        
        det_faces = clf_cad.detectMultiScale(face)
        total_seen += 1
        if len(det_faces) == 1:
            correct += 1
    print('face detection accuracy: {:.2f}%'.format(correct / total_seen * 100))
        