import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import torchvision.transforms.functional as tf
from PIL import Image
import random,cv2

import torch
import matplotlib.pyplot as plt
import numpy as np
from data.color_coding import color_coding
from data.augmentation import augmentation

class HairDataset(BaseDataset):
    """
    This dataset class can load datasets for hair image refinement.

    It requires two directories to host training images from domain A '/path/to/data/sketch',
    from domain M '/path/to/data/middle',
    and from domain B '/path/to/data/img' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(input_nc=3)
        parser.set_defaults(output_nc=3)
        parser.add_argument('--use_aug',action='store_true')
        parser.add_argument('--use_mean_color',action='store_true')
        parser.add_argument('--rotate_range',type=int,default=45)
        parser.add_argument('--sk_dir',type=str,default='')
        parser.add_argument('--matte_dir',type=str,default='')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        if opt.sk_dir != "":
             self.sk_dir = os.path.join(opt.dataroot, opt.sk_dir, opt.phase)
        else:
            self.sk_dir = os.path.join(opt.dataroot, 'sketch', opt.phase)
            
        if opt.matte_dir != "":
            self.matting_dir = opt.matte_dir
        else:
            self.matting_dir = os.path.join(opt.dataroot, 'matte', opt.phase)
        self.img_dir = os.path.join(opt.dataroot, 'img', opt.phase)  

        self.sk_paths = sorted(make_dataset(self.sk_dir, opt.max_dataset_size))   # load images from '/path/to/data/sketch/train'

        self.matting_paths = sorted(make_dataset(self.matting_dir, opt.max_dataset_size))
        self.img_paths = sorted(make_dataset(self.img_dir, opt.max_dataset_size))


        self.A_size = len(self.sk_paths)  # get the size of dataset A
        self.B_size = len(self.img_paths)
        assert self.A_size == self.B_size and self.A_size == len(self.matting_paths),  "A, matting and img directory size is not equal."


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, sk_paths and img_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            sk_paths (str)    -- image paths
            img_paths (str)    -- image paths
        """
        A_path = self.sk_paths[index % self.A_size]  # make sure index is within then range
        A_name = A_path.split('/')[-1]

        sk = cv2.imread(A_path,0)

        matting = cv2.imread(os.path.join(self.matting_dir,A_name))

        img = cv2.imread(os.path.join(self.img_dir,A_name))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.opt.use_mean_color:
            sk_mask = color_coding(img,sk,matting)
        else:
            sk_mask = color_coding(img,sk,matting,augment=self.opt.use_aug)

        if self.opt.use_aug:
            img, sk_mask, matting = augmentation(img,sk_mask,matting,rotate_range=[-self.opt.rotate_range, self.opt.rotate_range])

        h,w = img.shape[:2]
        noise = self.generate_noise(w,h)
        
        M = tf.to_tensor(matting)
        N = tf.to_tensor(noise)*2.0-1.0
        A = tf.to_tensor(sk_mask)*2.0-1.0
        B = tf.to_tensor(img)*2.0-1.0
        
        return {'A': A, 'B': B, 'N': N, 'M': M, 'A_paths': A_path}
    
    
            
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

    def generate_noise(self, width, height):
        weight = 1.0
        weightSum = 0.0
        noise = np.zeros((height, width, 3)).astype(np.float32)
        while width >= 8 and height >= 8:
            noise += cv2.resize(np.random.normal(loc = 0.5, scale = 0.25, size = (int(height), int(width), 3)), dsize = (noise.shape[0], noise.shape[1])) * weight
            weightSum += weight
            width //= 2
            height //= 2
        return noise / weightSum
