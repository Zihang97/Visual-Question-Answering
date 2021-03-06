## Architecture

<p align="center">
  <img src="picture/image1.png" width=500/>
</p>

FCN: Fully Convolutional Networks for Semantic Segmentation
</p>
CRF: Conditional Random Field
</p>
MRF: Markov random field
</p>
The front end uses FCN for rough feature extraction, and the back end uses CRF/MRF to optimize the output of the front end, and finally the segmentation map is obtained.
Next, we will summarize from the front-end and back-end parts.

## FCN

The network we use for classification usually connects several fully connected layers at the end. It squashes the original two-dimensional matrix (picture) into one-dimensional, thus losing spatial information, and finally trains to output a scalar. 
The output of image semantic segmentation needs to be a segmentation map, regardless of size, but at least two-dimensional. Therefore, we need to discard the fully connected layer and replace it with a fully convolutional layer, and this is a fully convolutional network.

### Convolutional

Convolution is to place ordinary classification networks, some VGG16, ResNet50/101 and other networks in fully connected layers, and replace them with corresponding convolutional layer.

<p align="left">
  <img src="picture/image5.png" width=400/>
</p>

### Upsample (Deconvolution)

Deconvolution is similar to convolution, both are operations of multiplication and addition. It's just that the latter is many-to-one, and the former is one-to-many. For the forward and backward propagation of deconvolution, only the forward and backward propagation of convolution can be reversed.

<p align="left">
  <img src="picture/image2.png" width=400/>
</p>

### Skip Layer

The function of this structure is to optimize the result, because if the result after full convolution is directly up-sampled, the result is very rough, so the results of different pooling layers must be up-sampled to optimize the output.

<p align="left">
  <img src="picture/image3.png" width=400/>
</p>

The results obtained by different upsampling structures are compared as follows:

<p align="left">
  <img src="picture/image4.png" width=400/>
</p>

### FCN Training Reproduce (CamVid)

<p align="center">
  <img src="picture/trials.png" width=1000/>
</p>

<p align="center">
  <img src="picture/result.gif" width=1000/>
</p>


## CRF
CRF is referred as Conditional Random Field, which is the most common-used back-end algorithm. 

<p align="center">
  <img src="picture/deeplab.PNG" width=600/>
</p>

The reason we need back-end algorithms is that convolutional network only output the value of each pixel independently, which don't care about the relationship between different pixels. So the using of CRF is to validate the edges of objects and make the areas smoothly.

<p align="center">
  <img src="picture/crf.PNG" width=150/>
</p>

Compared to MRF, CRF focuses on approaximating the conditional probability instead of joint probability, which is exactly what classification wants to do.

<p align="left">
  <img src="picture/bayesian.PNG" width=200/>
</p>

<p align="left">
  <img src="picture/KL.PNG" width=500/>
</p>

<p align="left">
  <img src="picture/crf_fomula.PNG" width=500/>
</p>

where p denotes position and I denotes color embeddings.


### Reproduce
Using 5 iterations of CRF

<p align="center">
  <img src="picture/im1.png" width=200>
  <img src="picture/out1.png" width=200>
</p>

<p align="center">
  <img src="picture/im2.png" width=200>
  <img src="picture/out2.png" width=200>
</p>

<p align="center">
  <img src="picture/im3.png" width=200>
  <img src="picture/out3.png" width=200>
</p>

## DeepLab
Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

<p align="center">
  <img src="picture/deeplab.png" >
  
## Sprint3
1. Combine the results from FCN and CRF, like deeplab

2. Develop our own dataset

3. Identify which area is background
