import os,sys,cv2,torch
from models.Unet_At_Bg import UnetAtGenerator
from models.Unet_At_Bg import UnetAtBgGenerator
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt
import numpy as np
from util import util

class Sk2Matte:
    def __init__(self,load_path="./checkpoints/S2M/200_net_G.pth"):
        self.model = UnetAtGenerator(1,1,8,64,use_dropout=True)
        self.device = torch.device('cuda:0')
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        model_name = '/'.join(load_path.split('/')[2:])
        print("Model Sk2Matte (%s) is loaded."%model_name)
    
    def getResult(self,inputs,img=None):
        inputs_tensor = tf.to_tensor(inputs[:,:,np.newaxis])*2.0-1.0
        inputs_tensor = inputs_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            result_tensor = self.model(inputs_tensor)
            result = ((result_tensor[0]+1)/2*255).cpu().numpy().transpose(1,2,0).astype("uint8")[...,0]
            # result = util.tensor2im(result_tensor)
            result[result>250] = 255

        return result

if __name__ == "__main__":
    ## You can prepare your own sketch for matte generation, i.e., hair strokes as 255, non-hair strokes as 0, background as 128

    # For unbraided hairstyle, run 'python S2I_test'
    # For braided hairstyle, run 'python S2I_test braid'
    if len(sys.argv) == 1:
        hair_type = 'unbraid' # default as unbraid_type
    else:
        hair_type = sys.argv[1]

    S2M = Sk2Matte()

    save_dir = "./results/generated_matte/"
    sk_dir = "./test_img/%s/input_1/"%hair_type
    sk_list = os.listdir(sk_dir)
    for i,path in enumerate(sk_list):
        print(i,path)
        sketch = cv2.imread(os.path.join(sk_dir,path),0)
        matte = S2M.getResult(sketch)
        cv2.imwrite(os.path.join(save_dir,path),matte)

        plt.subplot(121)
        plt.imshow(sketch,"gray")
        plt.subplot(122)
        plt.imshow(matte,"gray")
        plt.show()

