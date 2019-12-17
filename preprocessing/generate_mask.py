import os, sys
import cv2
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description="data preproc")

parser.add_argument("--mask_dir", type=str, default='./masks/')
parser.add_argument("--face_dir", type=str, default='../raw_datas/data1/')
parser.add_argument("--save_dir", type=str, default='./output/')
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--min_density", type=float, default=0.3)
parser.add_argument("--max_density", type=float, default=0.5)
parser.add_argument("--mask_scale_ratio", type=float, default=0.4)
parser.add_argument("--mask_darker_ratio", type=float, default=0.8)
parser.add_argument("--mask_tsp", type=float, default=0.4)
parser.add_argument("--face_scale_ratio", type=float, default=0.1)

args = parser.parse_args()
print(args)

mask_root = args.mask_dir
min_density = args.min_density
max_density = args.max_density
mask_scale_ratio = args.mask_scale_ratio
mask_darker_ratio = args.mask_darker_ratio
mask_tsp = args.mask_tsp
face_scale_ratio = args.face_scale_ratio


def scale_image(mask, ratio=1.0):
    w,h = mask.shape[:2]
    w = int(w * ratio)
    h = int(h * ratio)
    return cv2.resize(mask, (h, w))


def apply_transform(mask):
    # scaling
    scale_ratio = mask_scale_ratio + 0.2 * np.random.rand()
    mask = scale_image(mask, scale_ratio)

    # darker
    darker_ratio = mask_darker_ratio + 0.2 * np.random.rand()
    mask = np.asarray(mask, dtype=np.float32)
    mask *= darker_ratio

    # transpose
    if np.random.rand() > 0.5:
        mask = mask.transpose([1, 0, 2])

    # flip at axis 0
    if np.random.rand() > 0.5:
        mask = cv2.flip(mask, 0)

    # flip at axis 1
    if np.random.rand() > 0.5:
        mask = cv2.flip(mask, 1)

    return mask

def get_random_mask(w, h):

    mask_w, mask_h = 0, 0
    while(mask_w <= w or mask_h <= h):
        mask_id = random.randint(1,12)
        mask_path = os.path.join(mask_root, 'mask_{}.png'.format(mask_id))
        mask = cv2.imread(mask_path)
        
        # reset bg
        mask_sum = np.sum(mask, axis=2)
        mask[mask_sum == 255 * 3, :] = 0

        mask = apply_transform(mask)
        mask_w, mask_h, _ = mask.shape

    # print(mask_id)
    now_density = 0.0
    while(now_density < min_density or now_density > max_density):
        posx = random.randint(0, mask_w - w - 1)
        posy = random.randint(0, mask_h - h - 1)
        mask_selected = mask[posx:posx+w, posy:posy+h, :]
        mask_bin = np.sum(mask_selected, axis=2)
        now_density = np.sum(mask_bin > 0) / (mask_bin.shape[0] * mask_bin.shape[1])
    return mask_selected


def fuse_mask(img, mask):
    img_ = img.copy()
    mask_bin = np.sum(mask, axis=2)
    img_[mask_bin > 0, :] = mask_tsp * mask[mask_bin > 0, :] + (1 - mask_tsp) * img[mask_bin > 0, :]
    return img_


if __name__ == '__main__':

    faces_path = os.listdir(args.face_dir)
    faces_path = [os.path.join(args.face_dir, x) for x in faces_path]

    for path in faces_path:
        face = cv2.imread(path)
        print(path)
        if face is None:
            continue
        face = scale_image(face, face_scale_ratio)
        w, h, _ = face.shape

        for i in range(args.repeat):
            mask = get_random_mask(w, h)
            fused_face = fuse_mask(face, mask)
            save_path = path.split('/')[-1]
            save_path = '{}_{}.png'.format(save_path.split('.')[0], i)
            save_path = os.path.join(args.save_dir, save_path)
            cv2.imwrite(save_path, fused_face)
    print('done.')
