import os,cv2,random
import numpy as np
import matplotlib.pyplot as plt

def color_coding(img,sk_ori,mask,augment=False):
    sk_mask = np.array(mask)
    sk = np.array(sk_ori)

    while True:
        y, x  = np.where(sk!=0)
        
        if(len(y)==0):
            break
    
        temp_v = sk[y[0],x[0]]
        ys,xs = np.array(np.where(sk==temp_v))

        points_num = len(ys)
        
        if augment:
            use_mean = random.randint(0,5)
            if use_mean < 2:
                idx = random.randint(0,points_num-1)
                color = img[ys[idx],xs[idx]]
            else:
                color = (np.sum(img[ys,xs],axis=0)/points_num).astype("uint8")

        else:
            color = (np.sum(img[ys,xs],axis=0)/points_num).astype("uint8")
        sk_mask[ys,xs] = color
        sk[ys,xs] = 0

    return sk_mask
