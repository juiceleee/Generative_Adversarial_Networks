# Generative_Adversarial_Networks

Various GAN networks implemented with tensorflow/pytorch

* Pytorch Implementation: [**Yoo Jaehoon**](https://github.com/Ugness/)

* Tensorflow Implementation: [**juiceleee**](https://github.com/juiceleee/)

# Reference
* Vanilla_GAN : [**https://arxiv.org/abs/1406.2661**]

* DCGAN : [**https://arxiv.org/abs/1511.06434**]

* InfoGAN : [**https://arxiv.org/abs/1606.03657**]

# Requirements
```
opencv-python==3.3.0.10
Pillow==4.3.0
torch==0.4.0
tensorflow-gpu==1.7.0
tensorflow-tensorboard==0.4.0rc3
```

# Pytorch Network Architecture

Architectures are Different with Papers
  * Vanilla_GAN
    - D
      + Conv2d(1, 32, 3, 1), Relu
      + Conv2d(32, 64, 3, 1), Relu
      + FC(28x28x64, 625)
      + FC(625, 1)
      + dropout(0.5)
      + Sigmoid
    - G
      + FC(100, 256), Relu, BatchNorm
      + FC(256, 512), Relu, BatchNorm
      + FC(512, 28x28)
      + Sigmoid
  * DCGAN
    - D
      + Conv2d(1, 32, 3, 1), Relu
      + Conv2d(32, 64, 3, 1), Relu
      + FC(28x28x64, 625)
      + FC(625, 1)
      + dropout(0.5)
      + Sigmoid
    - G
      + FC(100, 7x7x16)
      + ConvTranspose2d(16, 4, 2, 2), Relu, BatchNorm
      + ConvTranspose2d(4, 1, 2, 2)
      + Sigmoid
  * InfoGAN
    - D_front
      + Conv2d(1, 32, 3, 1), LeakyRelu(0.1)
      + Conv2d(32, 64, 3, 1), LeakyRelu(0.1)
      + dropout(0.3)
      + FC(28x28x64, 625), LeakyRelu(0.1)
      - for D
        + D_front
        + FC(625, 1)
        + dropout(0.5)
        + Sigmoid
      - for Q_class
        + D_front
        + FC(625, 10), Softmax
      - for Q_cont
        + D_front
        + FC(625, 2), Sigmoid
    - G
      + FC(62+12, 7x7x16)
      + ConvTranspose2d(16, 4, 2, 2), Relu, BatchNorm
      + ConvTranspose2d(4, 1, 2, 2)
      + Sigmoid
