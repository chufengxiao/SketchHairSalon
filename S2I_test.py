
import os,sys,cv2,torch
from models.Unet_At_Bg import UnetAtGenerator
from models.Unet_At_Bg import UnetAtBgGenerator
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt
import numpy as np
from util import util

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
    # For unbraided hairstyle, run 'python S2I_test'
    # For braided hairstyle, run 'python S2I_test braid'
    if len(sys.argv) == 1:
        hair_type = 'unbraid' # default as unbraid_type
    else:
        hair_type = sys.argv[1]

    if hair_type == "unbraid":
        S2I = Sk2Image("./checkpoints/S2I_unbraid/200_net_G.pth")
    else:
        S2I = Sk2Image("./checkpoints/S2I_braid/400_net_G.pth")
    
    sk_dir = "./test_img/%s/input_2/"%hair_type
    matte_dir = "./test_img/%s/matte"%hair_type
    img_dir = "./test_img/%s/img/"%hair_type

    save_dir = "./results/generated_%s"%hair_type

    os.makedirs(save_dir,exist_ok = True)
    sk_list = os.listdir(sk_dir)
    for i,path in enumerate(sk_list):
        print(i,path)
        sketch = cv2.imread(os.path.join(sk_dir,path))
        matte = cv2.imread(os.path.join(matte_dir,path))
        img = cv2.imread(os.path.join(img_dir,path))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        sk_matte = np.array(matte)
        sk_rgb = cv2.cvtColor(sketch,cv2.COLOR_BGR2RGB)
        sk_gray = cv2.cvtColor(sketch,cv2.COLOR_BGR2GRAY)
        sk_matte[sk_gray!=0]=sk_rgb[sk_gray!=0]
        result = S2I.getResult(sk_matte,img,matte)

        cv2.imwrite(os.path.join(save_dir,path),cv2.cvtColor(result,cv2.COLOR_RGB2BGR))

        # Show the input and output
        plt.subplot(141)
        plt.imshow(sk_rgb,"gray")
        plt.subplot(142)
        plt.imshow(matte)
        plt.subplot(143)
        plt.imshow(img)
        plt.subplot(144)
        plt.imshow(result)
        plt.show()

