from torch_model import *
import torch
import cv2
from PIL import Image
import numpy as np
import os

np.set_printoptions(threshold=np.nan, linewidth=np.nan)
# from tensorflow.examples.tutorials.mnist import input_data

D_front, D, Q, G = torch.load('models/05_04_23_55/Epoch_70.pt')
noise_n = 62

raw_noise = make_noise(10, noise_n)
# raw_noise = torch.randn([10, noise_n])
noise = torch.Tensor(raw_noise).cuda()
noise_code = make_noise(10, 2)
noise_code = torch.Tensor(noise_code).cuda()
G = G.cuda()
noise.requires_grad_(False)
G.eval()
i = 1
for j in range(10):
    results = G(noise, one_hot([j for _ in range(10)]), noise_code)
    for result in results:
        os.makedirs('results', exist_ok=True)
        result = torch.reshape(result, (28, 28))
        result = result.detach().cpu().numpy()
        result = result*256//1
        print(result)
        os.makedirs('results/{}'.format(j), exist_ok=True)
        cv2.imwrite("results/{}/{}.jpg".format(j, i), result)
        # im = Image.fromarray(result, 'L')
        # im.save("results/{}.jpg".format(i))
        i += 1
