from torch_model import Discriminator, Generator, make_noise
import torch
import cv2
from PIL import Image
import numpy as np

np.set_printoptions(threshold=np.nan, linewidth=np.nan)
# from tensorflow.examples.tutorials.mnist import input_data

D1 = torch.load("../models/05_01_00_36/D_53.pt", map_location=lambda storage, loc: storage)
G1 = torch.load("../models/05_01_00_36/G_53.pt", map_location=lambda storage, loc: storage)

noise_n = 100

# raw_noise = make_noise(100, noise_n)
raw_noise = torch.randn([100, noise_n])
noise = torch.Tensor(raw_noise).cuda()
G1 = G1.cuda()
noise.requires_grad_(False)
G1.eval()
i = 1
# print(raw_noise)
results = G1(noise)
# torchvision.utils.save_image(torch.unsqueeze(results, 1), "results/{}.jpg".format(i))
# im = Image.open('results/individualImage.png')
# print(np.reshape(np.mean(np.array(im.getdata(), np.uint8), 1), (28, 28)))

for result in results:
    result = torch.reshape(result, (28, 28))
    result = result.detach().cpu().numpy()
    result = result*256//1
    print(result)
    cv2.imwrite("../results/{}.jpg".format(i), result)
    # im = Image.fromarray(result, 'L')
    # im.save("results/{}.jpg".format(i))
    i += 1
