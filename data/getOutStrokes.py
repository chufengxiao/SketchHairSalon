#%%
import os,cv2,random
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.ndimage import morphology

def getEdge(mask):
    edge = np.zeros_like(mask)
    dst1 = 0 #np.random.randint(0,5) # 0
    dst2 = dst1+3 #np.random.randint(3,12) # 3
    
    out_mask = 1 - mask
    out_edt = morphology.distance_transform_edt(out_mask)
    unknown_inside = (out_edt <= dst2) * (out_edt > dst1)

    edge[unknown_inside ] = 255

    return edge

def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    
    mask = np.zeros((h, w), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)

    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        # brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        brushWidth = maxBrushWidth
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask

def generate_stroke_mask(im_size, parts=5, maxVertex=20, maxLength=100, maxBrushWidth=50, maxAngle=180):
    mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)

    return mask

def blend_outStroke(sketch,outStrokes):
    sketch_mask = np.array(sketch,dtype=np.bool)
    outStrokes_mask = np.array(outStrokes,dtype=np.bool)
    img = np.ones_like(sketch) * 128
    img[sketch_mask] = 255
    img[outStrokes_mask] = 0
    return img

def ran_getEdge(matte,sketch):

    mask = np.array(matte,dtype=np.bool).astype("uint8")
    edge = getEdge(mask)

    ran_mask = generate_stroke_mask(mask.shape[:2])
    outStrokes = edge*ran_mask

    # edge2 = getEdge(mask)
    # ran_mask2 = generate_stroke_mask(mask.shape[:2])
    # outStrokes2 = edge2*ran_mask2
    # outStrokes = outStrokes+outStrokes2
    # outStrokes[outStrokes>=255]=255

    inputs = blend_outStroke(sketch,outStrokes)
    return inputs





if __name__ == "__main__":
    matte_dir = "/home/chufengxiao/Hair_Synthesis/datasets/IG/exp12_matting/matting/train"
    sk_dir = "/home/chufengxiao/Hair_Synthesis/datasets/IG/exp12_matting/sketch/train"
    img_dir = "/home/chufengxiao/Hair_Synthesis/datasets/IG/exp12_matting/img/train"
    matte_list = ['CM_3.png',"CM_77.png","CM_69.png","CM_170.png","CM_222.png","CM_291.png","CM_341.png","CM_343.png","CM_451.png","CM_477.png","CM_661.png","R2_540.png","R2_550.png"]
    edge_aug = Edge_Aug()
    for i,path in enumerate(matte_list):
        path = "CM_77.png"
        print(i,path)

        matte = cv2.imread(os.path.join(matte_dir,path),0)
        sk = cv2.imread(os.path.join(sk_dir,path),0)
        img = cv2.imread(os.path.join(img_dir,path))
        edge_aug.getInputs(sk,matte,img)
        # inputs = ran_getEdge_0402(matte,sketch)
        

        

#%%