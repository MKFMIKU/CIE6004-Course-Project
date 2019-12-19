import numpy as np
import cv2
import os, sys
from tqdm import tqdm


def scale_image(mask, ratio=1.0):
    w,h = mask.shape[:2]
    w = int(w * ratio)
    h = int(h * ratio)
    return cv2.resize(mask, (h, w))


def convert_bg(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(hsv[0,0,:])

    lower_blue = np.array([90, 170, hsv[0,0,2]-50])
    upper_blue = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    erode = cv2.erode(mask, None, iterations=1)
    dilate=cv2.dilate(erode, None, iterations=1)

    mask_opp = np.sum(dilate) / dilate.shape[0] / dilate.shape[1] / 255
    # print(mask_opp)
    if mask_opp < 0.4 or mask_opp > 0.48:
        return None

    img_ = img.copy()
    img_[dilate == 255, :] = 255
    return img_


if __name__ == '__main__':

    blue_dir = '../raw_datas/blue_data3_2/'
    save_dir = '../raw_datas/data3_2/'
    faces_path = os.listdir(blue_dir)

    # # test_path = './1020320116.jpg'
    # # test_path = './1010510521.jpg'
    # test_path = './1012310117.jpg'
    # # test_path = './1012310221.jpg'
    # face = cv2.imread(test_path)
    # white_face = convert_bg(face)
    # cv2.imwrite('test_out.png', white_face)

    for path in tqdm(faces_path):
        read_path = os.path.join(blue_dir, path)
        face = cv2.imread(read_path)
        if face is None:
            continue
        white_face = convert_bg(face)

        save_path = path.split('.')[0] + '.jpg'
        save_path = os.path.join(save_dir, save_path)
        if white_face is not None:
            cv2.imwrite(save_path, white_face)
        
