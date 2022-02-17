import cv2,os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import squeeze
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize
from skimage import measure as skim

def remove_small_regions(edge_img, small_cc):
    filtered_edges = np.zeros_like(edge_img, dtype=bool)

    # Find connected regions and remove small components
    ## ps: skim measure label aims to return the the resgions which its neighbors has the samller memebers than the threshold
    rgb_edge_regions, num_regions = skim.label(edge_img, return_num=True, connectivity=2)
    for rid in range(1, num_regions + 1):
        rid_region = (rgb_edge_regions == rid)
        if np.sum(np.array(rid_region, dtype=np.int32)) > small_cc:
            filtered_edges = np.logical_or(rid_region, filtered_edges)

    return filtered_edges

def visualSketch(input_strokes, matte, added_strokes):
    matte = matte[:,:,np.newaxis]
    visual = np.concatenate((matte,matte,matte),axis=2)

    added_strokes = np.array(added_strokes,dtype=np.bool)
    visual[added_strokes] = [19,193,14]

    input_strokes = np.array(input_strokes,dtype=np.bool)
    visual[input_strokes] = [38,135,208]

    return visual

def getSketchCompletion(sk_gray,matte,vis_flag=False):
    
    sk_mask = 0 + (sk_gray > 0)

    mask = np.array(matte)
    mask = 0 + (mask > 230)

    dist_mask = distance_transform_edt(mask)
    # Step 1: remove soft boundary
    squeeze_mask = (dist_mask > 10).astype("uint8")
    
    dist_sk_mask = distance_transform_edt(1-sk_mask)
    filter_im = dist_sk_mask * squeeze_mask
    threshold = 15

    # Step 2: subtract the existing-stroke regions from the generated matte
    filter_regions = (filter_im > threshold).astype(np.uint8)

    # Step 3: remove very small regions
    refined_regions = remove_small_regions(filter_regions, small_cc=240)

    # Step 4: extract skeleton (strokes) from the rest regions
    skeleton = skeletonize(np.uint8(refined_regions), method='lee')
    dist_skeleton = distance_transform_edt(1-skeleton)
    added_stroke = (dist_skeleton < 1.5) * 255

    sk_matte_full = visualSketch(sk_mask,matte,added_stroke)
    if vis_flag:
        plt.subplot(231)
        plt.imshow(visualSketch(sk_mask,matte,sk_mask))
        plt.subplot(232)
        plt.imshow(squeeze_mask)
        plt.subplot(233)
        plt.imshow(filter_regions)
        plt.subplot(234)
        plt.imshow(refined_regions)
        plt.subplot(235)
        plt.imshow(added_stroke)
        plt.subplot(236)
        plt.imshow(sk_matte_full)
        plt.show()
    
    return added_stroke, sk_matte_full

def getUnbraidStroke(sketch,matte,getVisual=False):
    sk_gray = cv2.cvtColor(sketch,cv2.COLOR_RGB2GRAY)

    added_stroke, sk_matte_full = getSketchCompletion(sk_gray,matte,vis_flag=getVisual)

    sk_matte_incom = np.zeros_like(sketch)+matte[:,:,np.newaxis]
    sk_matte_incom[sk_gray!=0]=[38,135,208]

    return added_stroke


if __name__ == '__main__':
    sketch = cv2.imread('./incom_unbraid_sk.png')
    matte = cv2.imread('./incom_unbraid_matte.png',0)

    # set getVisual as True and you can see the visualization of unbraid auto-completion process
    getUnbraidStroke(sketch,matte,getVisual=True)