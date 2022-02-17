import cv2,os
from mayavi import mlab
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation
from scipy.ndimage.morphology import distance_transform_edt

pi = np.pi

def fish_tail(t,width,b=10,w=1,r=5,use_noise = False):
    strand_list = []
    radius = r*width
    a = width
    s_num = 5

    if use_noise:
        noise = 1+np.random.normal(0,0.1,s_num)
    else:
        noise = np.ones((s_num,))

    for i in range(s_num):
        x = a * np.sin(w*t+2*i*pi/s_num)
        y = t * 20
        z = b * np.sin(2*(w*t+2*i*pi/s_num))
        r = noise[i]*radius
        strand_list.append([x,y,z,r])
    strand_list = np.array(strand_list)
    return strand_list

def strand_2(t,width,b=20,w=1,r=5):
    
    strand_list = []
    radius = r*width*1.4
    a = width
    s_num = 3

    x = a * np.sin(w*t) 
    y = t * 20
    z = b * np.sin((w*t))
    r = radius
    strand_list.append([x,y,z,r])

    x = a * np.sin(w*t+pi) 
    y = t * 20
    z = b * np.sin((w*t+0.75*pi))
    r = radius
    strand_list.append([x,y,z,r])
    
    strand_list = np.array(strand_list)
    return strand_list

def strand_3(t,width,b=10,w=1,r=5,use_noise = True):
    strand_list = []
    radius = r*width
    a = width
    s_num = 3

    if use_noise:
        noise = 1+np.random.normal(0,0.1,s_num)
    else:
        noise = np.ones((s_num,))

    for i in range(s_num):
        x = a * np.sin(w*t+i*2*pi/3)
        y = t * 20
        z = b * np.sin(2*(w*t+i*2*pi/3))
        r = noise[i]*radius
        strand_list.append([x,y,z,r])
    
    strand_list = np.array(strand_list)
    return strand_list

def strand_4(t,width,b=10,w=1,r=5,use_noise=False):
    def f(t):
        n_pi = t // pi
        c2 = (n_pi%2 == 0)

        t1 = np.sin(4.0*t)
        t2 = np.sin(2.0*t)
        res = t1

        res[c2]=t2[c2]
        
        return res
        
    strand_list = []
    radius = r*width
    a = width
    s_num = 4

    if use_noise:
        noise = 1+np.random.normal(0,0.1,s_num)
    else:
        noise = np.ones((s_num,))

    for i in range(s_num):

        x = a * np.sin(w*t+i*pi/2)
        y = t
        z = b * f(w*t+i*pi/2)
        r = noise[i]*radius
        strand_list.append([x,y,z,r])
    
    strand_list = np.array(strand_list)
    return strand_list

def strand_5(t,width,b=10,w=1,r=5,use_noise=False):
    strand_list = []
    radius = r*width
    a = width
    s_num = 5
    
    if use_noise:
        noise = 1+np.random.normal(0,0.1,s_num)
    else:
        noise = np.ones((s_num,))

    for i in range(s_num):
        x = a * np.sin(w*t+2*i*pi/s_num)
        y = t
        z = b * np.sin(4*(w*t+2*i*pi/s_num))
        r = noise[i]*radius
        strand_list.append([x,y,z,r])
        
    strand_list = np.array(strand_list)
    return strand_list

def getEdge(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(img,10,10)

    dist_edge = distance_transform_edt(255-edge)
    edge = (dist_edge <= 1) * 1
    return edge

def plot3D_line(strand_list,colors = [[38,135,208],[255,144,144],[165,249,163],[255,235,59],[83,195,204]]):
    bg_color = (0, 0, 0)
    colors = np.array(colors,dtype="uint8")/255
    figure=mlab.figure(1, bgcolor=bg_color, fgcolor=(0.5, 0.5, 0.5),size=(512,564))
    ends = np.array([[0,511],[0,511],[0,0],[50,50]],dtype="int")
    out = mlab.points3d(ends[0],ends[1],ends[2],ends[3],color=bg_color,scale_factor=1)
    out.actor.property.lighting = False

    for i,item in enumerate(strand_list):
        out=mlab.points3d(item[0],item[1],item[2],item[3],color=tuple(colors[i]),scale_factor=.25,resolution=16,line_width=1)

        out.actor.property.lighting = False

    mlab.view(azimuth=180, elevation=0, distance=-950, focalpoint=(255.5,255.5,0))

    f = mlab.gcf()
    f.scene._lift()

    img = mlab.screenshot()

    mlab.orientation_axes()
    mlab.axes()
    mlab.clf(figure)
    mlab.close()
    return img

def transBraid(strand_list,sk_list=None):
    shift_x,shift_y = sk_list[0], sk_list[1]
    strand_list[:,0,:] = (strand_list[:,0,:]+shift_x)
    strand_list[:,1,:] = shift_y

    return strand_list

def getBoundaries(sk):
    sk = np.array(sk,dtype=np.bool).astype("uint8")
    dist_sk = distance_transform_edt(sk)
    sk = (dist_sk > 1).astype("uint8")

    sk, contours, hierarchy = cv2.findContours(sk,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 2:
        return None # There are not two boundaries
    else:
        contours = contours[:2]
        l_list = []
        for i in contours:
            l = np.array(i).squeeze().T # l(x,y) with shape of (2,num)
            l_list.append(l)
        return l_list,sk

def rotateHLines(sk):
    # If the lines are horizontal path then rotate it to be vertical for convenient computation

    l_list, _ = getBoundaries(sk)
    if l_list is None:
        return None, None
    else:
        line = l_list[0]
        t,b = np.min(line[1]),np.max(line[1])
        l,r = np.min(line[0]),np.max(line[0])

        if b-t < r-l:
            isH = True
            sk = cv2.rotate(sk, cv2.ROTATE_90_CLOCKWISE)
        else:
            isH = False

    return sk, isH


def getSkList(sk):
    def getEnds(l0,l1):
        y0,y1 = l0[1],l1[1]
        t0,b0 = np.min(y0),np.max(y0)
        t1,b1 = np.min(y1),np.max(y1)

        top = max(t0,t1)
        bottom = min(b0,b1)

        return top,bottom
    
    l_list, sk = getBoundaries(sk)

    t,b = getEnds(l_list[0],l_list[1])

    l0 = []
    l1 = []
    for y in range(t,b+1):
        x = np.where(sk[y,:]==1)[0]
        l0.append([x[0],y])
        l1.append([x[-1],y])

    l0 = np.array(l0).T
    l1 = np.array(l1).T
    trim_sk_list = [l0,l1]
    return trim_sk_list

def plotSk(sk_list,thick=3):
    sk = np.zeros((512,512),dtype="uint8")
    for stroke in sk_list:
        x,y = stroke
        for i in range(len(y)-1):
            cv2.line(sk,(x[i],y[i]),(x[i+1],y[i+1]),255,thick)
    return sk

def getGlobalStructure(sk_list):
    l1, l2 = sk_list
    len_1, len_2 = len(l1[0]), len(l2[0])

    if len_1 > len_2:
        l1, l2 = l2, l1
        len_1, len_2 = len_2, len_1
    sample_idx = (np.arange(0,len_1,1)*(len_2/len_1)).astype("int")
    l2 = l2[:,sample_idx]

    medial = ((l1+l2)/2).astype("int")

    dist = np.power(np.sum(np.power((medial - l1),2),axis=0),0.5)

    return medial, dist

class Braids:
    def __init__(self):
        self.styles = {'1':fish_tail,'2':strand_2,'3':strand_3,'4':strand_4,'5':strand_5}

    def completion(self,sk,s_num=3,w=1,colors = [[38,135,208],[255,144,144],[165,249,163],[255,235,59],[83,195,204]]):
        '''
        Params: 
        1. 'sk' is the input braided sketch with only one channel;
        2. 's_num' indicates one braide type, i.e., {'1':fish_tail,'2':strand_2,'3':strand_3,'4':strand_4,'5':strand_5};
        3. 'w' indicates the direction and number of braided knots;
        4. 'colors' indicates the colors assigned to each strand tube as a color palette. Here the default colors are set for visualization if runing this script. If running the interface, the colors are assigned by users.
        '''
        sk, isH = rotateHLines(sk)
        if sk is None:
            return None

        s_num = str(s_num)
        l_list = getSkList(sk)

        trans_line, dist = getGlobalStructure(l_list)
        r = 5
        d_y = 0.05
        end = len(trans_line[1])*d_y
        t = np.arange(0,end,d_y,dtype="float")

        a_width = dist / 1.75 # a_width:real_width = 1:1.75
        t = t[:len(a_width)]
 
        strand_list = self.styles[s_num](t,width=a_width,w=w,r=r)
        strand_list = transBraid(strand_list,trans_line)

        img = plot3D_line(strand_list)

        edge = getEdge(img)

        color_p = self.styles[s_num](t,width=a_width,w=w,r=r*1.4)
        strand_list = transBraid(color_p,trans_line)
        color_palette = plot3D_line(strand_list,colors=colors)

        color_edge = color_palette * edge[:,:,np.newaxis]

        if isH:
            color_edge = cv2.rotate(color_edge,cv2.ROTATE_90_COUNTERCLOCKWISE)
        return color_edge
        

if __name__ == "__main__":
    sk = cv2.imread("./incom_braid_bound.png",0)
    h,w = sk.shape[:2]

    braid_model = Braids()
    comp_sk = braid_model.completion(sk,s_num=3,w=1)
    if comp_sk is None:
        print('The method cound not detect two boundaries, please make sure you draw correctly!')

    plt.subplot(121)
    plt.imshow(sk)
    plt.subplot(122)
    plt.imshow(comp_sk)
    plt.show()
