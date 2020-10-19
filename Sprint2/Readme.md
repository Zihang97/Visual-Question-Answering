# Architecture

# FCN

# CRF
CRF is referred as Conditional Random Field, which is the most common-used back-end algorithm.

<p align="center">
  <img src="picture/crf.PNG" width=150/>
</p>

The reason we need back-end algorithms is that convolutional network only output the value of each pixel independently, which don't care about the relationship between different pixels. So the using of CRF is to validate the edges of objects and make the areas smoothly.

<p align="left">
  <img src="picture/crf_fomula.PNG" width=500/>
</p>

where p denotes position and I denotes color embeddings.

<p align="center">
  <img src="picture/deeplab.PNG" width=600/>
</p>

## Reproduce

<p align="left">
  <img src="picture/im1.png" width=150>
  <img src="picture/out1.png" width=150>
</p>

# DeepLab

# Sprint3
