# Model Summeries
Here are the full model summeries for each part of the VAE model. To verify yourself, simply run:
```
from torchsummary import summary
model = VAEModel(base=16).to(device)
summary(model.encoder,(512,1,1))
summary(model.encoder,(1,256,256))
```

## Encoder:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1         [-1, 16, 128, 128]             144
       BatchNorm2d-2         [-1, 16, 128, 128]              32
         LeakyReLU-3         [-1, 16, 128, 128]               0
              Conv-4         [-1, 16, 128, 128]               0
            Conv2d-5         [-1, 32, 128, 128]           4,608
       BatchNorm2d-6         [-1, 32, 128, 128]              64
         LeakyReLU-7         [-1, 32, 128, 128]               0
              Conv-8         [-1, 32, 128, 128]               0
            Conv2d-9           [-1, 32, 64, 64]           9,216
      BatchNorm2d-10           [-1, 32, 64, 64]              64
        LeakyReLU-11           [-1, 32, 64, 64]               0
             Conv-12           [-1, 32, 64, 64]               0
           Conv2d-13           [-1, 32, 64, 64]           9,216
      BatchNorm2d-14           [-1, 32, 64, 64]              64
        LeakyReLU-15           [-1, 32, 64, 64]               0
             Conv-16           [-1, 32, 64, 64]               0
           Conv2d-17           [-1, 32, 32, 32]           9,216
      BatchNorm2d-18           [-1, 32, 32, 32]              64
        LeakyReLU-19           [-1, 32, 32, 32]               0
             Conv-20           [-1, 32, 32, 32]               0
           Conv2d-21           [-1, 64, 32, 32]          18,432
      BatchNorm2d-22           [-1, 64, 32, 32]             128
        LeakyReLU-23           [-1, 64, 32, 32]               0
             Conv-24           [-1, 64, 32, 32]               0
           Conv2d-25           [-1, 64, 16, 16]          36,864
      BatchNorm2d-26           [-1, 64, 16, 16]             128
        LeakyReLU-27           [-1, 64, 16, 16]               0
             Conv-28           [-1, 64, 16, 16]               0
           Conv2d-29           [-1, 64, 16, 16]          36,864
      BatchNorm2d-30           [-1, 64, 16, 16]             128
        LeakyReLU-31           [-1, 64, 16, 16]               0
             Conv-32           [-1, 64, 16, 16]               0
           Conv2d-33             [-1, 64, 8, 8]          36,864
      BatchNorm2d-34             [-1, 64, 8, 8]             128
        LeakyReLU-35             [-1, 64, 8, 8]               0
             Conv-36             [-1, 64, 8, 8]               0
           Conv2d-37           [-1, 1024, 1, 1]       4,195,328
        LeakyReLU-38           [-1, 1024, 1, 1]               0
----------------------------------------------------------------
Total params: 4,357,552
Trainable params: 4,357,552
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.25
Forward/backward pass size (MB): 36.14
Params size (MB): 16.62
Estimated Total Size (MB): 53.01
----------------------------------------------------------------
```

## Decoder:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 1024, 1, 1]         525,312
   ConvTranspose2d-2             [-1, 64, 8, 8]       4,194,304
       BatchNorm2d-3             [-1, 64, 8, 8]             128
         LeakyReLU-4             [-1, 64, 8, 8]               0
     ConvTranspose-5             [-1, 64, 8, 8]               0
            Conv2d-6             [-1, 64, 8, 8]          36,864
       BatchNorm2d-7             [-1, 64, 8, 8]             128
         LeakyReLU-8             [-1, 64, 8, 8]               0
              Conv-9             [-1, 64, 8, 8]               0
  ConvTranspose2d-10           [-1, 64, 16, 16]          65,536
      BatchNorm2d-11           [-1, 64, 16, 16]             128
        LeakyReLU-12           [-1, 64, 16, 16]               0
    ConvTranspose-13           [-1, 64, 16, 16]               0
           Conv2d-14           [-1, 64, 16, 16]          36,864
      BatchNorm2d-15           [-1, 64, 16, 16]             128
        LeakyReLU-16           [-1, 64, 16, 16]               0
             Conv-17           [-1, 64, 16, 16]               0
  ConvTranspose2d-18           [-1, 64, 32, 32]          65,536
      BatchNorm2d-19           [-1, 64, 32, 32]             128
        LeakyReLU-20           [-1, 64, 32, 32]               0
    ConvTranspose-21           [-1, 64, 32, 32]               0
           Conv2d-22           [-1, 32, 32, 32]          18,432
      BatchNorm2d-23           [-1, 32, 32, 32]              64
        LeakyReLU-24           [-1, 32, 32, 32]               0
             Conv-25           [-1, 32, 32, 32]               0
  ConvTranspose2d-26           [-1, 32, 64, 64]          16,384
      BatchNorm2d-27           [-1, 32, 64, 64]              64
        LeakyReLU-28           [-1, 32, 64, 64]               0
    ConvTranspose-29           [-1, 32, 64, 64]               0
           Conv2d-30           [-1, 32, 64, 64]           9,216
      BatchNorm2d-31           [-1, 32, 64, 64]              64
        LeakyReLU-32           [-1, 32, 64, 64]               0
             Conv-33           [-1, 32, 64, 64]               0
  ConvTranspose2d-34         [-1, 32, 128, 128]          16,384
      BatchNorm2d-35         [-1, 32, 128, 128]              64
        LeakyReLU-36         [-1, 32, 128, 128]               0
    ConvTranspose-37         [-1, 32, 128, 128]               0
           Conv2d-38         [-1, 16, 128, 128]           4,608
      BatchNorm2d-39         [-1, 16, 128, 128]              32
        LeakyReLU-40         [-1, 16, 128, 128]               0
             Conv-41         [-1, 16, 128, 128]               0
  ConvTranspose2d-42         [-1, 16, 256, 256]           4,096
      BatchNorm2d-43         [-1, 16, 256, 256]              32
        LeakyReLU-44         [-1, 16, 256, 256]               0
    ConvTranspose-45         [-1, 16, 256, 256]               0
           Conv2d-46          [-1, 1, 256, 256]             145
          Sigmoid-47          [-1, 1, 256, 256]               0
----------------------------------------------------------------
Total params: 4,994,641
Trainable params: 4,994,641
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 69.26
Params size (MB): 19.05
Estimated Total Size (MB): 88.31
----------------------------------------------------------------
```
