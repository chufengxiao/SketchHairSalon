import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.getOutStrokes import ran_getEdge
import torchvision.transforms.functional as tf
from PIL import Image
import random,cv2
from data.getOutStrokes import getEdge, generate_stroke_mask, blend_outStroke
from data.augmentation import augmentation

import torch
import matplotlib.pyplot as plt
import numpy as np

class MatteDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(input_nc=1)
        parser.set_defaults(output_nc=1)
        parser.add_argument('--no_edgeStroke', action='store_true', help='do not add edgeStroke')
        parser.add_argument('--use_aug', action='store_true')
        parser.add_argument('--no_outEdge', action='store_true')
        parser.add_argument('--inputs_dir', type=str, default="")

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot,'unbraid', 'sketch', opt.phase)
        self.dir_B = os.path.join(opt.dataroot, 'unbraid', 'matte', opt.phase) 

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.dir_A = os.path.join(opt.dataroot,'braid', 'sketch', opt.phase)
        self.dir_B = os.path.join(opt.dataroot, 'braid', 'matte', opt.phase)
        self.A_paths += sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
        self.B_paths += sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)

        assert self.A_size == self.B_size, "A and B size is not equal."
        self.edge_aug = Edge_Aug()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.A_size]  # make sure index is within then range
        A_name = A_path.split('/')[-1]
        B_name = B_path.split('/')[-1]
        
        assert A_name == B_name, "A path %s and B %s path are not matched."%(A_name, B_name)

        sketch = cv2.imread(A_path,0)
        matte = cv2.imread(B_path,0)

        if self.opt.inputs_dir == "":
            inputs, matte = self.edge_aug.getInputs(sketch,matte,use_aug=self.opt.use_aug,no_outEdge=self.opt.no_outEdge)
        else:
            inputs = cv2.imread(os.path.join(self.opt.inputs_dir,A_name),0)
        
        A = tf.to_tensor(inputs[:,:,np.newaxis])*2.0-1.0
        B = tf.to_tensor(matte[:,:,np.newaxis])*2.0-1.0
        
        return {'A': A, 'B': B, 'A_paths': A_path}
               
    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]

        return filename1_without_ext == filename2_without_ext

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size


class Edge_Aug:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))

    def ran_getEdge_0402(self,matte,sketch):
        mask = np.array(matte,dtype=np.bool).astype("uint8")
        
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel,iterations=1)
        edge = getEdge(closed)

        ran_mask = generate_stroke_mask(mask.shape[:2],parts=6)
        outStrokes = edge*ran_mask

        inputs = blend_outStroke(sketch,outStrokes)
    
        
        return inputs

    def getInputs(self,sk,matte,use_aug=True,no_outEdge=False):

        sk = np.array(sk,dtype=np.bool).astype("uint8")*255

        if use_aug:
            img,sk,matte = augmentation(sk,sk,matte)
            sk = np.array(sk,dtype=np.bool).astype("uint8")*255
        
        if no_outEdge:
            edge = blend_outStroke(sk,np.zeros_like(sk))
        else:
            edge = self.ran_getEdge_0402(matte,sk)

        return edge,matte