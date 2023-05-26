# Image Colorization 

Image colorization is the process of adding color to grayscale or black and white images. It involves converting the intensity values of the grayscale image into a color space, like RGB, Lab etc. and then filling in the missing color channels to create a full-color image. Various techniques can be used for image colorization, including image processing methods like segmentation and texture synthesis, as well as machine learning approaches.

Ab approach to image colorization involves using Generative Adversarial Networks (GANs). In this setup, a generator network generates the color version of the grayscale image, while a discriminator network is trained to distinguish between the generated color version and real color images. The generator network improves its colorization capabilities by trying to fool the discriminator network.

## Generative adversarial networks (GAN)

A GAN is a special type of deep learning network that can create new data that looks similar to the data it was trained on.

A GAN consist of two networks that train together:
* Generator - The Generator network takes random numbers as input and creates new data that has the same structure as the training data
* Discriminator - The discriminator network receives batches of data that include both real observations from the training data and generated data from the generator. Its job is to classify these observations as either "real" or "generated". 
!!! IMG HERE





```python
import torch
```

## References

Inspiration, code snippets, etc.
* [ddd]()

## dasd:
* []()

