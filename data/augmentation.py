import os,cv2,random
import numpy as np
import matplotlib.pyplot as plt

def augmentation(img,sk,mask,rotate_range=[-15,15],rest_imgs=[]):
    
    use_flip = random.randint(0,1)
    use_rotate = random.randint(0,5)
    use_trans = random.randint(0,5)

    if use_flip == 1:
        img,sk,mask,rest_imgs = flip(img,sk,mask,rest_imgs)

    if use_rotate >= 1:
        img,sk,mask,rest_imgs = rotate_image(img,sk,mask,rotate_range,rest_imgs)

    if use_trans >= 1:
        img,sk,mask,rest_imgs = translate_image(img,sk,mask,rest_imgs)

    if len(rest_imgs)!=0:
        return img,sk,mask,rest_imgs
    else:
        return img,sk,mask

def flip(img, sk, mask, rest_imgs):
    img = cv2.flip(img,1)
    mask = cv2.flip(mask,1)
    sk = cv2.flip(sk,1)

    rest_flip = []
    if len(rest_imgs) != 0:
        for i in rest_imgs:
            rest_flip.append(cv2.flip(i,1))
    
    return img,sk,mask,rest_flip

def rotate_image(img, sk, mask, range, rest_imgs):
    angle = random.randint(range[0],range[1])
    M_rotation = cv2.getRotationMatrix2D((256, 256), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotation, (512, 512),flags=cv2.INTER_NEAREST)
    sk_rotated = cv2.warpAffine(sk, M_rotation, (512, 512),flags=cv2.INTER_NEAREST)
    mask_rotated = cv2.warpAffine(mask, M_rotation, (512, 512),flags=cv2.INTER_NEAREST)

    rest_rotated = []
    if len(rest_imgs) != 0:
        for i in rest_imgs:
            rest_rotated.append(cv2.warpAffine(i, M_rotation, (512, 512)))

    return img_rotated, sk_rotated, mask_rotated, rest_rotated

def translate_image(img, sk, mask, rest_imgs):
    c_is_3 = False
    if len(mask.shape) == 3:
        c_is_3 = True
        mask = mask[:,:,0]

    r, c = np.where(mask!=0)
    top, bottom = np.min(r), np.max(r)
    left, right = np.min(c), np.max(c)

    r_shift = random.randint(-top,512-bottom)
    c_shift = random.randint(-left,512-right)

    mat_shift = np.float32([[1,0,c_shift], [0,1,r_shift]])
    img = cv2.warpAffine(img, mat_shift, (512, 512),flags=cv2.INTER_NEAREST)
    sk = cv2.warpAffine(sk, mat_shift, (512, 512),flags=cv2.INTER_NEAREST)
    mask = cv2.warpAffine(mask, mat_shift, (512, 512),flags=cv2.INTER_NEAREST)

    rest_trans = []
    if len(rest_imgs) != 0:
        for i in rest_imgs:
            rest_trans.append(cv2.warpAffine(i, mat_shift, (512, 512)))

    if c_is_3:
        mask_1 = mask[:,:,np.newaxis]
        mask = np.concatenate((mask_1,mask_1,mask_1),axis=2)
    return img, sk, mask, rest_trans


    