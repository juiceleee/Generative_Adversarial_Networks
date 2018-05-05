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
  * Vanilla_GAN
    - D
      + Conv2d(1, 32, 3, 1), Relu
      + Conv2d(32, 64, 3, 1), Relu
      + FC(28*28*64, 625)
      + FC(625, 1)
      + dropout(0.5)
      + Sigmoid
    - G
      + FC(100, 256), Relu, BatchNorm
      + FC(256, 512), Relu, BatchNorm
      + FC(512, 28*28), Sigmoid
  * DCGAN
  * InfoGAN
