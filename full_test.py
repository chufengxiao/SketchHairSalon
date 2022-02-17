
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

class Sk2Image:
    def __init__(self,load_path="./checkpoints/S2I_unbraid/200_net_G.pth"):
        self.model = UnetAtBgGenerator(3,3,8,64,use_dropout=True)
        self.device = torch.device('cuda:0')
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        model_name = '/'.join(load_path.split('/')[2:])
        print("Model Sk2Image (%s) is loaded."%model_name)
    
    def getResult(self,inputs,img,matte):
            
        h,w = img.shape[:2]
        noise = self.generate_noise(w,h)

        N = tf.to_tensor(noise)*2.0-1.0
        N = N.unsqueeze(0).to(self.device)

        img_tensor = tf.to_tensor(img)*2.0-1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        M = tf.to_tensor(matte).unsqueeze(0).to(self.device)

        inputs = tf.to_tensor(inputs)*2.0-1.0
        inputs = inputs.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            result_tensor = self.model(inputs,img_tensor, M,N)
            result = ((result_tensor[0]+1)/2*255).cpu().numpy().transpose(1,2,0).astype("uint8")
        
        return result

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

if __name__ == "__main__":
    if len(sys.argv) == 1:
        hair_type = 'unbraid' # default as unbraid_type
    else:
        hair_type = sys.argv[1]

    S2M = Sk2Matte()
    if hair_type == "unbraid":
        S2I = Sk2Image("./checkpoints/S2I_unbraid/200_net_G.pth")
    else:
        S2I = Sk2Image("./checkpoints/S2I_braid/400_net_G.pth")
    sk_1_dir = "./test_img/%s/input_1/"%hair_type
    sk_2_dir = "./test_img/%s/input_2/"%hair_type
    img_dir = "./test_img/%s/img/"%hair_type

    sk_list = os.listdir(sk_1_dir)

    save_1_dir = "./results/generated_matte/"
    save_2_dir = "./results/generated_%s"%hair_type

    os.makedirs(save_1_dir,exist_ok = True)
    os.makedirs(save_2_dir,exist_ok = True)

    for i,path in enumerate(sk_list):
        print(i,path)
        input_1 = cv2.imread(os.path.join(sk_1_dir,path),0)
        input_2 = cv2.imread(os.path.join(sk_2_dir,path))
        img = cv2.imread(os.path.join(img_dir,path))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        matte = S2M.getResult(input_1)

        matte_3 = cv2.cvtColor(matte,cv2.COLOR_GRAY2RGB)
        sk_matte = np.array(matte_3)
        sk_rgb = cv2.cvtColor(input_2,cv2.COLOR_BGR2RGB)
        sk_gray = cv2.cvtColor(input_2,cv2.COLOR_BGR2GRAY)
        sk_matte[sk_gray!=0]=sk_rgb[sk_gray!=0]

        result = S2I.getResult(sk_matte,img,matte_3)

        cv2.imwrite(os.path.join(save_1_dir,path),input_1)
        cv2.imwrite(os.path.join(save_2_dir,path),cv2.cvtColor(result,cv2.COLOR_RGB2BGR))

        # plt.subplot(131)
        # plt.imshow(input_1,"gray")
        # plt.subplot(132)
        # plt.imshow(sk_matte)
        # plt.subplot(133)
        # plt.imshow(result)
        # plt.show()

