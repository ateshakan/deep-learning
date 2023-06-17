# Image Colorization Colorization with conditional GANs

Image colorization refers to the process of adding
color to grayscale or black and white images. This is an
important task in the field of computer vision, as it allows us to
restore the vividness and vibrancy of old images and make them
more appealing to the viewer. The process of image colorization
can be challenging for several reasons. Firstly, it is a subjective
task, as different people may have different opinions about what
colors should be used for a given grayscale image. Secondly, the
process can be time-consuming and labor-intensive, especially
when performed manually. Finally, there are also technical
challenges involved, such as color consistency and preserving
the details in the image. Recently, deep learning techniques
have been applied to the task of image colorization, which
has led to significant improvements in the quality of colorized
images. However, there are still challenges to be overcome,
such as improving the realism of the generated colors and
handling complex scenes with multiple objects and diverse color
distributions.

## **1. Introduction**

Traditional methods for image colorization can be divided
into two categories: manual and semi-automated methods.
Manual methods involve a human artist manually adding
color to the gray-scale image, either using digital tools such
as a graphics tablet or by hand-painting the image. While
these methods can produce high-quality results, they are time consuming and labor-intensive, making them unsuitable for
large-scale image colorization tasks. Semi-automated methods
are designed to reduce the time and effort required for manual
colorization. These methods typically involve using a set
of pre-defined color palettes or a set of rules to automatically colorize the image. For example, some methods use a
combination of color histograms and texture information to
determine the colors for different regions in the image.

While these methods are faster than manual methods, they
are often limited by the quality of the pre-defined color
palettes or the accuracy of the rules used. In both manual
and semi-automated methods, it can be difficult to achieve
color consistency across the entire image, and preserving the
details and textures in the image can also be a challenge.
In addition, these methods may struggle to handle complex
scenes with multiple objects and diverse color distributions.
Furthermore, traditional methods can often produce results
that look unrealistic or unnatural, as they may not be able
to accurately capture the true colors and shades present in the
image. Another limitation of traditional methods is that they
do not take into account any context or information about the
scene in the image, which can lead to inaccurate colorization
results. For example, traditional methods may not be able to
distinguish between different types of objects, such as skin,
hair, and clothing, and therefore may use the same colors for
all of these objects, even though they have different color
distributions in reality.

Overall, while traditional methods have made some
progress in the field of image colorization, they still face
several challenges and limitations that can limit their
effectiveness and accuracy. In recent years, deep learning
methods have emerged as a promising alternative for image
colorization, as they are capable of learning the complex
relationships between the grayscale input and the desired
color output, leading to improved results.

###  **U-net Autoencoder**:
An autoencoder is a type of neural
network that is designed to learn a compact representation of
the input data, and then use this representation to reconstruct
the original data. Autoencoders can be used for various tasks,
including image colorization. The U-net architecture is a
specific type of autoencoder that is commonly used for image
colorization. The Unet architecture consists of two parts: an
encoder and a decoder.

The encoder part of the network is responsible for learning a compact representation of the input grayscale image,
while the decoder part is responsible for reconstructing the
original image from this representation. In the case of image
colorization, the Unet autoencoder is trained on a large dataset
of grayscale and color images. During training, the encoder
part of the network is trained to learn a representation of the
grayscale image that contains information about the colors in
the image.

The decoder part of the network is then trained to generate
the corresponding color image from this representation. The
training process involves minimizing a reconstruction loss,
which measures the difference between the generated color
image and the ground-truth color image. This process is
repeated for many training examples, allowing the network
to learn the relationship between the grayscale input and the
desired color output.

The results achieved using U-net autoencoders for image
colorization have been promising. U-net autoencoders have
been shown to produce colorized images that are visually
appealing and have improved color consistency compared
to traditional methods. In addition, U-net autoencoders can
handle complex scenes with multiple objects and diverse
color distributions and are capable of generating plausible
colors for a wide range of images. However, there are also
limitations to the U-net autoencoder approach. For example,
the generated colors may still look unrealistic or unnatural,
and there may beissues with preserving the details and
textures in the image. In addition, the U-net autoencoder
approach may not be suitable for images with very low
resolution or for images with complex color distributions.

### **GAN:**

Generative Adversarial Networks (GANs) are a type
of deep learning architecture that are designed to generate
new, synthetic data that resembles the input data. GANs
consist of two main components: a generator network and a
discriminator network. The generator network is responsible
for generating new samples of the desired data distribution,
while the discriminator network is responsible for evaluating
the authenticity of the generated samples. The generator and
discriminator networks are trained in a competition, with the
generator trying to generate samples that the discriminator
cannot differentiate from the real data, and the discriminator
trying to accurately identify whether a sample is real or
generated.

For image colorization, a GAN can be used by treating the
grayscale image as the input to the generator network and the
corresponding color image as the target. During training, the
generator network is trained to generate a colorized version of
the grayscale image, while the discriminator network is trained
to distinguish between the generated color image and the
ground-truth color image. The training process involves iteratively updating both the generator and discriminator networks,
with the generator network trying to generate increasingly
realistic color images and the discriminator network trying
to identify the generated images with increasing accuracy.

The results achieved using GANs for image colorization
have been promising. GANs have been shown to produce
colorized images that are visually appealing and have improved color consistency compared to traditional methods.
In addition, GANs are capable of handling complex scenes
with multiple objects and diverse color distributions and can
generate plausible colors for a wide range of images. However,
there are also limitations to the GAN approach. For example,
the generated colors may still look unrealistic or unnatural,
and there may be issues with preserving the details and
textures in the image. In addition, the GAN approach requires
a large amount of training data and may be difficult to train
due to the potential for mode collapse, where the generator
network only generates a limited range of outputs.

Our specific scenario involves a conditional GAN, where
the generator takes grayscale images as input and aims to
produce colorized versions of those images. Instead of using
random noise as input, we provide the grayscale image as a
condition to the generator. We base our implementation on
the Pix2Pix conditional GAN model. The generator learns a
function through training that transforms the input grayscale
image into a colorized version. The discriminator then compares this output image with the corresponding ground truth
color image from the dataset, classifying it as either real or
generated.

This approach allows us to generate realistic and visually
appealing colorized images based on the given grayscale inputs


### **Lab color space:** 
The LAB color space is often preferred
over RGB for image colorization because it separates the
image’s luminance information (lightness) from its color information (chrominance) in a way that is more perceptually
uniform. This means that changes made to the A and B
channels in LAB space will correspond more closely to
changes in human perception of color than changes made to
the RGB channels. Additionally, the LAB color space has a
wider gamut of colors than RGB, meaning it can represent
more colors accurately, especially in the blue-green range.

By using the LAB color space instead of the RGB color
space, image processors can achieve more accurate and subtle
color shifts, resulting in more realistic and appealing colorizations.

The CIE Lab (Lightness, A, B) color space is a color model
designed to approximate human perception of color. It is one
of the color spaces defined by the International Commission
on Illumination (CIE). Unlike RGB color model, which is
device-dependent, CIE Lab is device-independent and based
on the human visual system.

CIE Lab color space consists of three components: L, a,
and b. Here’s a brief description of each component:
* **L (Lightness):** It represents the lightness or brightness
of the color. The L component ranges from 0 to 100,
where 0 represents black and 100 represents white. The
value of L increases as the color gets lighter.
* **a (Green-Red axis):** It represents the position between
green and red in the color spectrum. The a component
ranges from -128 to +127, with negative values indicating
greenness and positive values indicating redness.
* **b (Blue-Yellow axis):** It represents the position between
blue and yellow in the color spectrum. The b component
ranges from -128 to +127, with negative values indicating
blueness and positive values indicating yellowness.

By using Lab color space, we feed only the L channel
(grayscale image) to the model and hope it will predict the
other two channels (a and b) so that we can concatenate all
channels and get a colored image. However, in RGB, we
would have first to convert the image to grayscale, and then
give that to the model, and hope that the model will predict
3 numbers for each pixel, which is a far more difficult and
unstable task due to the many more possible combinations of
3 numbers compared to two numbers.

In CIE Lab color space, the color information is separated
from the luminance information, making it a suitable color
space for color difference calculations and color perception
studies. It is often used in color management systems, image
processing algorithms, and applications where accurate color
reproduction and color differences are important.

Conversion between RGB and CIE Lab color spaces is
possible using specific mathematical formulas defined by
the CIE. The conversion allows transforming colors between
different color spaces for various color-related operations.

### Dataset: CIFAR-10
CIFAR-10 dataset consists of 60000 32x32 color images in
10 classes, with 6000 images per class. There are 50000
training images and 10000 test images. The colorization
pipeline implemented in our project involves several stages to
transform grayscale images into realistic and vibrant colored
images. The pipeline begins with dataset preparation, where
we gather a dataset consisting of grayscale images paired
with their corresponding RGB counterparts. These image pairs
are then converted from RGB to the CieLAB color space to
facilitate efficient colorization.

## **Results**








```python
import torch
```

## References

Inspiration, code snippets, etc.
* [ddd]()

## dasd:
* []()

