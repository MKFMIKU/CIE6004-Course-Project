import numpy as np
import cv2
import os, sys

def pre_img(demo, demo2):
    rets = demo.copy()
    mask = cv2.cvtColor(demo.copy(),cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(mask, 225, 255,cv2.THRESH_BINARY_INV)
    cv2.imwrite('temp_bin.jpg', thresh)

    thresh, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.drawContours(thresh, contours, -1, (0,255,0), 8)
    cv2.imwrite('temp_cont.jpg', thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('temp_thresh.png', thresh)
    retss = rets.copy()
    retss[thresh == 0, :] = 255
    cv2.imwrite('temp_thresh_test.jpg', retss)

    
    white_bg = np.ones_like(demo) * 255
    ImageOne = cv2.bitwise_and(demo, demo, mask = thresh)
    white_bg = cv2.bitwise_not(white_bg, white_bg, mask = thresh)
    demo = ImageOne+white_bg
    mask = cv2.cvtColor(demo.copy(), cv2.COLOR_RGB2GRAY)

    demo2[thresh == 0, :] = 0
    mask_demo2 = cv2.cvtColor(demo2.copy(), cv2.COLOR_RGB2GRAY)
    ret2, thresh2 = cv2.threshold(mask_demo2, 60, 255, cv2.THRESH_BINARY_INV)
    demo[thresh2 > 0, :] = 255
    rets[thresh2 > 0, :] = 255
    print(thresh2.shape)

    r, g, b = cv2.split(demo)

    ret,thresh_g =  cv2.threshold(g, 125, 255, cv2.THRESH_BINARY_INV)
    ret,thresh_r = cv2.threshold(mask.copy(), 100, 255, cv2.THRESH_BINARY_INV)

    thresh_g_r = thresh_g-thresh_r

    batch_masks = cv2.cvtColor(thresh_g_r,cv2.COLOR_GRAY2RGB)
    batch_masks = cv2.normalize(batch_masks.astype('float'),None,0.,1.,cv2.NORM_MINMAX)
    batch_images = cv2.normalize(demo.astype('float'),None,-1.,1.,cv2.NORM_MINMAX)

    return batch_masks,batch_images,demo,demo2,thresh2,rets



face_path = 'example_mask.jpg'
face2_path = 'example_mask_2.jpg'
face = cv2.imread(face_path)
face2 = cv2.imread(face2_path)
batch_masks, batch_images, output, output2, thresh2, ret = pre_img(face, face2)
cv2.imwrite('rmbg_test2.jpg', output)
cv2.imwrite('rmgb_output2.jpg', output2)
cv2.imwrite('rmgb_thresh2.png', thresh2)
cv2.imwrite('rmgb_ret.jpg', ret)
cv2.imwrite('batch_masks.jpg', batch_masks * 255)
cv2.imwrite('batch_images.jpg', batch_images * 255)

print(batch_masks.shape)
output[batch_masks == 1] = 255
cv2.imwrite('rmbg_test3.jpg', output)

